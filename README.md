# Anonymous ARR June 2024 Submission

This repository contains the code for the paper: **"Benchmarking Emotion Recognition with Vision-Language Models"**. It introduces an evaluation of popular Vision-Language Models (VLMs) like GPT4-omni, LLaVA, LLaVA-Next, and Qwen-VL on a novel benchmark, **EvE**, for <ins>Ev</ins>oked <ins>E</ins>motion Recognition. **EvE** is created by including several important datasets for evoked emotion recognition, which are: EmoSet, FI, ArtPhoto, Abstract and Emotion6. Further, as EmoSet and FI contain a significantly large number of samples, to downsize them, while maintaining a significant difficulty level of the samples chosen, a finetuning-based strategy is chosen. 

The 'src' directory contains the main code for the evaluation process, which includes a dataset setup code, and the evaluation for individual models. The 'benchmark_processing' directory contains the code used to finetune a ViT model to obtain difficult samples for inclusion in the benchmark, from the datasets of EmoSet, and FI. The evaluation proceeds in 4 different modes, namely **classification** (simple multimodal prompting), **explanation** (multimodal prompting while self-explaining the generations), **context-based reasoning** (where the VLMs self-reason in a chain-of-thought manner, about the contextual information present in images), and **caption-based reasoning** (where the VLMs perform emotional reasoning based on the captions they generate for the images). 

To run an evaluation of any of the VLMs on any of the subdatasets in **EvE**, run the following: 


`python3 -d "name of the dataset" -m "name of the model" -e "name of evaluation mode"`

where `-d` takes the following options: 

- emoset
- fi
- abstract
- artphoto
- emotion6

`-m` takes the options: 

- gpt (Referring to GPT4-omni)
- llava
- llavanext
- qwen-vl

and `-e` takes the options: 

- classification
- explanation
- context-reasoning
- caption-reasoning


