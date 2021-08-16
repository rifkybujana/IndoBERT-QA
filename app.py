# Copyright 2021 Rifky Bujana Bisri & Muhammad Fajrin Buyang Daffa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import streamlit as st
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import Trainer

from datasets import Dataset

import LoggingSet
LoggingSet.set_global_logging_level(LoggingSet.logging.ERROR)


page = st.sidebar.selectbox("Page", ["Teman Belajar", "About"])

st.title(f"*{page}*")

# model path
model_checkpoint = "model"

@st.cache(allow_output_mutation=True, show_spinner=False)
def load():
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    trainer = Trainer(model=model)

    return tokenizer, trainer

with st.spinner("Loading Model..."):
    tokenizer, trainer = load()

pad_on_right = tokenizer.padding_side == "right"
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.


def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 100):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            print (f'\n{start_logits[cls_index]}')

            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            sorted_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
            best_answer = sorted_answer[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer
        answer = best_answer["text"] if abs(best_answer["score"]) - abs(min_null_score) > -1 else "there is no such information in the text"
        predictions[example["id"]] = answer

    return predictions, sorted_answer


def predict(context, question):
    data = []
    if isinstance(question, list):
        for i, q in enumerate(question):
            data_temp = {
                'id': i,
                'context': context,
                'question': q
            }
            data.append(data_temp)
    else:
        data.append({
            'id': 0,
            'context': context,
            'question': question
        })

    data = Dataset.from_pandas(pd.DataFrame(data))
    data_feature = data.map(prepare_validation_features, batched=True, remove_columns=data.column_names)
    
    raw_prediction = trainer.predict(data_feature)
    final_predictions, answers = postprocess_qa_predictions(data, data_feature, raw_prediction.predictions)

    return final_predictions, answers
    
if page == "Teman Belajar":
    context = st.text_area(label="Context", help="Teks yang ingin dianalisa", height=200)
    question = st.text_input(label="Question", help="Pertanyaan yang ingin ditanyakan terkait teks")

    if st.button("Submit"):
        with st.spinner("Computing..."):
            predictions, answers = predict(context, question)
            
        st.success(predictions[0])
        with st.beta_expander("Other Answer"):
            st.write([{
                'score': answer['score'].item(),
                'text': answer['text']
            } for answer in answers])
else:
    with open("README.md", "r") as f:
        st.write("[![Star](https://img.shields.io/github/stars/rifkybujana/IndoBERT-QA.svg?logo=github&style=social)](https://gitHub.com/rifkybujana/IndoBERT-QA) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/rifkybujana/IndoBERT-QA/blob/main/LICENSE)")
        st.write(f.read())