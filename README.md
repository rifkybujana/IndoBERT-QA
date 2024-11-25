**Notice of Attribution Clarification**

This is to clarify that **Muhammad Fajrin Buyang Daffa** is not, and has never been, a part of this project. They have made no contributions to this repository, and as such, shall not be given any attribution in relation to this work.

For further inquiries, please contact the rifky@genta.tech.

> This project is part of my research who won a silver medal at the KoPSI (Kompetisi Penelitian Siswa Indonesia/Indonesian Student Research Competition) entitled "Teman Belajar: Asisten Digital Pelajar SMAN 28 Jakarta dalam Membaca".
## indoBERT Base-Uncased fine-tuned on Translated Squad v2.0
[IndoBERT](https://huggingface.co/indolem/indobert-base-uncased) trained by [IndoLEM](https://indolem.github.io/) and fine-tuned on [Translated SQuAD 2.0](https://github.com/Wikidepia/indonesian_datasets/tree/master/question-answering/squad) for **Q&A** downstream task.

**Model Size** (after training): 420mb

## Details of indoBERT (from their documentation)
[IndoBERT](https://huggingface.co/indolem/indobert-base-uncased) is the Indonesian version of BERT model. This model trained using over 220M words, aggregated from three main sources:
- Indonesian Wikipedia (74M words)
- news articles from Kompas, Tempo (Tala et al., 2003), and Liputan6 (55M words in total)
- an Indonesian Web Corpus (Medved and Suchomel, 2017) (90M words).

This model trained for 2.4M steps (180 epochs) with the final perplexity over the development set being 3.97 (similar to English BERT-base).
This IndoBERT was used to examine IndoLEM - an Indonesian benchmark that comprises of seven tasks for the Indonesian language, spanning morpho-syntax, semantics, and discourse.
## Details of the downstream task (Q&A) - Dataset
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

| Dataset  | Split | # samples |
| -------- | ----- | --------- |
| SQuAD2.0 | train | 130k      |
| SQuAD2.0 | eval  | 12.3k     |
## Model Training
The model was trained on a Tesla T4 GPU and 12GB of RAM.
## Results:
| Metric | # Value   |
| ------ | --------- |
| **EM** | **51.61** |
| **F1** | **69.09** |
## Pipeline Usage
```py
from pipeline import Pipeline

pipeline = Pipeline("model")
pipeline.predict(
    "Pangeran Harya Dipanegara (atau biasa dikenal dengan nama Pangeran Diponegoro, lahir di Ngayogyakarta Hadiningrat, 11 November 1785 – meninggal di Makassar, Hindia Belanda, 8 Januari 1855 pada umur 69 tahun) adalah salah seorang pahlawan nasional Republik Indonesia, yang memimpin Perang Diponegoro atau Perang Jawa selama periode tahun 1825 hingga 1830 melawan pemerintah Hindia Belanda. Sejarah mencatat, Perang Diponegoro atau Perang Jawa dikenal sebagai perang yang menelan korban terbanyak dalam sejarah Indonesia, yakni 8.000 korban serdadu Hindia Belanda, 7.000 pribumi, dan 200 ribu orang Jawa serta kerugian materi 25 juta Gulden.",
    "kapan pangeran diponegoro lahir?"
)
```
*output:*
```py
{
    "best answer": OrderedDict([(0, '11 November 1785')]),
    "answers": [{'score': 12.36208, 'text': '11 November 1785'}, {'score': 9.136721, 'text': '11 November 1785 - meninggal di Makassar, Hindia Belanda, 8 Januari 1855'}, {'score': 8.018387, 'text': '1785'}, {'score': 6.1863337, 'text': 'Ngayogyakarta Hadiningrat, 11 November 1785'}, {'score': 6.091961, 'text': '11 November 1785 -'}, {'score': 5.8137712, 'text': '11 November 178'}, {'score': 5.579988, 'text': '11 November'}, {'score': 5.423601, 'text': '11 November 1785 - meninggal di Makassar, Hindia Belanda, 8 Januari 1855 pada umur 69 tahun'}, ...]
}
```
## Simple Usage (Using Huggingface)
```py
from transformers import pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="Rifky/Indobert-QA",
    tokenizer="Rifky/Indobert-QA"
)
qa_pipeline({
    'context': """Pangeran Harya Dipanegara (atau biasa dikenal dengan nama Pangeran Diponegoro, lahir di Ngayogyakarta Hadiningrat, 11 November 1785 – meninggal di Makassar, Hindia Belanda, 8 Januari 1855 pada umur 69 tahun) adalah salah seorang pahlawan nasional Republik Indonesia, yang memimpin Perang Diponegoro atau Perang Jawa selama periode tahun 1825 hingga 1830 melawan pemerintah Hindia Belanda. Sejarah mencatat, Perang Diponegoro atau Perang Jawa dikenal sebagai perang yang menelan korban terbanyak dalam sejarah Indonesia, yakni 8.000 korban serdadu Hindia Belanda, 7.000 pribumi, dan 200 ribu orang Jawa serta kerugian materi 25 juta Gulden.""",
    'question': "kapan pangeran diponegoro lahir?"
})
```
*output:*
```py
{
  'answer': '11 November 1785',
  'end': 131,
  'score': 0.9272009134292603,
  'start': 115
}
```
### Reference
Fajri Koto and Afshin Rahimi and Jey Han Lau and Timothy Baldwin. 2020. IndoLEM and IndoBERT: A Benchmark Dataset and Pre-trained Language Model for Indonesian NLP. Proceedings of the 28th COLING.
