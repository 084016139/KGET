import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers.activations import ACT2FN
from datasets import load_dataset, load_metric,Dataset,DatasetDict
# model_checkpoint = "fnlp/bart-base-chinese"
# model_checkpoint = "./outputs"
# model_checkpoint = "./outputforner"
model_checkpoint = "./outputkgdone"
# raw_datasets = load_dataset("wmt16", "ro-en")
#
# raw_datasets.save_to_disk('this')
# print(raw_datasets)
dataset = Dataset.from_csv('6sxmx.csv')
vadataset = Dataset.from_csv('7sxmx.csv')
# dataset = Dataset.from_csv('kgu.csv')
# vadataset = Dataset.from_csv('kgu.csv')
datasets = DatasetDict({'train':dataset,'test':dataset,'validation':vadataset})
print(datasets)
# DatasetDict['train'] = Dataset.from_csv('s.csv')
# metric = load_metric("sacrebleu")
# print(metric)
# raw_datasets.to_csv('s.csv')
#
# dataset = Dataset.from_csv('s.csv')
# print(type(dataset))
# print(type(raw_datasets['train']))
#
# # for i in dataset:
# #
# #     s = eval(i["translation"])
# #     print(s)
# #     print(s['en'])
# #     print(s['tr'])
#
#
#     # print(i["translation"].split("'en': '")[1].split("', 'tr': '")[1])
#
max_input_length = 500
max_target_length = 500
# source_lang = "tr"
# target_lang = "en"
prefix = ""

# metric = load_metric("bleu")
metric = load_metric("sacrebleu")
# raw_datasets = Dataset.from_csv('s.csv')
from transformers import AutoTokenizer

# 需要安装`sentencepiece`： pip install sentencepiece

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


if "mbart" in model_checkpoint:
    tokenizer.src_lang = "zh"
    tokenizer.tgt_lang = "zh"

with tokenizer.as_target_tokenizer():
    print(tokenizer("左下肺癌脑转移,复查彩超又见胰腺占位,提示转移可能,近两周来,昏沉多寐,体重下降6斤,偶有头疼,肝区胀痛,性情抑郁,易怒,后背发痒,口苦,疲劳乏力,大便2－3日一行,尿黄"))
    model_input = tokenizer("左下肺癌脑转移,复查彩超又见胰腺占位,提示转移可能,近两周来,昏沉多寐,体重下降6斤,偶有头疼,肝区胀痛,性情抑郁,易怒,后背发痒,口苦,疲劳乏力,大便2－3日一行,尿黄")
    tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'])
    # 打印看一下special toke
    print('tokens: {}'.format(tokens))
def preprocess_function(examples):

    # inputs = [prefix + eval(ex)[source_lang] for ex in examples["translation"]]
    # targets = [eval(ex)[target_lang] for ex in examples["translation"]]

    inputs = [prefix + ex.split("&*")[0].replace('"',"") for ex in examples["translation"]]
    targets = [ex.split("&*")[1].replace('"',"") for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # print(inputs,targets)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    # print(model_inputs["labels"])
    return model_inputs
tokenized_datasets = datasets.map(preprocess_function, batched=True)



from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,BartForConditionalGeneration

# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from model_bart3 import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
# BartForConditionalGeneration.forward()
batch_size = 1
args = Seq2SeqTrainingArguments(
    "test-translation",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=False,
    generation_num_beams=3,
    # save_strategy="no",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # num_beams=100
    # output = False
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(result)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()

# text = "代诉：肺癌脑部多发性转移,11月21日上海华山医院γ刀治疗,面色萎黄欠华,腰酸隐痛,两侧胁肋时有痛感,活动不利,食纳尚可,咳嗽,痰粘色黄"
# def genee(text):
#
#
#
#     inputs = tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=max_input_length,
#             return_tensors="pt",
#         )
#     input_ids = inputs.input_ids.to(model.device)
#     translation = model.generate(input_ids)
#     result = tokenizer.batch_decode(translation, skip_special_tokens=True)
#     return result[0]
# with open('3432.txt', 'r') as f:
#     data = f.readlines()
# with open('444.txt','w') as f:
#     for i in data:
#         f.write(i.replace("\n","")+"===>"+ge nee(i)+"\n")
#         print(i.replace("\n","")+"===>"+genee(i)+"\n")
#

trainer.save_model('outputkg')


