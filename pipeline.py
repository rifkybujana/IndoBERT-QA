"""
    Copyright 2021 Rifky Bujana Bisri & Muhammad Fajrin Buyang Daffa

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import collections
import datasets
import torch

import numpy as np
import pandas as pd

from datasets import Dataset
from datasets.utils.logging import set_verbosity_error

from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

datasets.disable_progress_bar()
set_verbosity_error()

class Pipeline:
    """
    Utility to build prepare model and predict question answering task

    Args:
        - model_checkpoint: path to the model located (local path or huggingface path)
        - max_length: The maximum length of a feature (question and context)
        - doc_stride: The authorized overlap between two part of the context when splitting it is needed.
        - impossible_answer: make this model to predict if the question is related to the context or not
    """

    def __init__(self, model_checkpoint="Rifky/Indobert-QA", max_length=384, doc_stride=128, impossible_answer=False):
        self.__model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

        self.__tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.__MAX_LENGTH = max_length
        self.__DOC_STRIDE = doc_stride

        self.__PAD_ON_RIGHT = self.__tokenizer.padding_side == "right"
        self.__impossible_answer = impossible_answer

    def __preprocess(self, data):
        """
        Before we can feed those texts to our model, we need to preprocess them. 
        This is done by a 🤗 Transformers Tokenizer which will (as the name indicates) 
        tokenize the inputs (including converting the tokens to their corresponding IDs 
        in the pretrained vocabulary) and put it in a format the model expects, as 
        well as generate the other inputs that model requires.

        Args:
            - data: data that you want to feed into model

        Output:
            - tokenized_examples: data that has been preprocessed
        """

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        data["question"] = [question.lstrip() for question in data["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.__tokenizer(
            data["question" if self.__PAD_ON_RIGHT else "context"],
            data["context" if self.__PAD_ON_RIGHT else "question"],
            truncation="only_second" if self.__PAD_ON_RIGHT else "only_first",
            max_length=self.__MAX_LENGTH,
            stride=self.__DOC_STRIDE,
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
            context_index = 1 if self.__PAD_ON_RIGHT else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(data["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def __postprocess(
        self, raw_data, features, raw_predictions, n_best_size=10, max_answer_length=100
    ):
        """
        Postprocess the output of the model into readable text and score

        Args:
            - raw_data: the data we want to predict before taking into any process
            - features: data that already preprocessed
            - raw_predictions: output of the model prediction
            - n_best_size: number of best answer we want to consider as predictions
            - max_answer_length: maximum answer of this model

        Output:
            - predictions: best answer
            - answers: list of all the model answer
        """

        all_start_logits, all_end_logits = raw_predictions

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(raw_data["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Let's loop over all the examples!
        for example_index, example in enumerate(raw_data):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if impossible_answer is True.
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
                cls_index = features[feature_index]["input_ids"].index(
                    self.__tokenizer.cls_token_id
                )
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
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
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "text": context[start_char:end_char],
                            }
                        )

            if len(valid_answers) > 0:
                answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
                best_answer = answers[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer
            if not self.__impossible_answer:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = (
                    best_answer["text"] if best_answer["score"] > min_null_score else ""
                )
                predictions[example["id"]] = answer

        return {
            "best answer": predictions,
            "answers": answers
        }

    def predict(self, context, questions):
        """
        Predict the answer of a question of a context

        Args:
            - context: context of the question
            - question: question that we want the model to answer

        output:
            - Best answer
            - List of answer
        """

        data = []

        # Prepare all the question to be processed
        if isinstance(questions, list):
            for i, question in enumerate(questions):
                data_temp = {"id": i, "context": context, "question": question}
                data.append(data_temp)
        else:
            data.append({"id": 0, "context": context, "question": questions})

        # Convert data into dataset to make it faster and easier to process
        data = Dataset.from_pandas(pd.DataFrame(data))
        # Process the data
        data_feature = data.map(
            self.__preprocess,
            batched=True,
            remove_columns=data.column_names,
        )
        
        temp_data_feature = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for i in data_feature:
            for k, v in i.items():
                if k in temp_data_feature.keys():
                    temp_data_feature[k].append(v)

        # Get model prediction
        raw_prediction = self.__model(**{k: torch.tensor(v) for k, v in temp_data_feature.items()})
        del temp_data_feature

        # Return final prediction and list of all model answer
        return self.__postprocess(
            data, data_feature, [raw_prediction.start_logits.detach().numpy(), raw_prediction.end_logits.detach().numpy()]
        )
