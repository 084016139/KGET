import numpy
from transformers import AutoModel, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
# model_checkpoint = "fnlp/bart-large-chinese"
# model_checkpoint = "./output1"
# model_checkpoint = "./outputner"
model_checkpoint = "./outputkg"
import torch
from model_bart3 import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
# model = AutoModel.from_pretrained(model_checkpoint)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# text = "去年底胸痛,咯痰带血,目前苦胸痛,咳嗽,咯痰有时带血,夜晚口干苦,动则气喘，"
# text = "代诉：住院化疗稍恶心呕吐,厌食,右侧腹胀,上腹有肿胀感,化疗后大便排出鲜血,量不多,下肢肿，"
# text = '自觉头晕昏胀疼痛，视物模糊，动则恶心呕吐，左手麻，月经绝经5年，'
# text = '左上肺鳞癌手术化疗后,最近咳减不尽,痰色白质稠,气短,两肩关节疼痛,两下肢浮肿,膝冷,大便偏烂,'
# text = "代诉：化疗，恶心，呕吐,纳差,右侧腹胀,上腹胀,大便带鲜血,血量不多,下肢肿"
# text = "咳减痰不多,口时干,胃不胀,大便日2－3次,偏软"
text = '两月来语声低微,胸部闷塞,吐痰色白或黄,偶夹血丝,肘膝关节酸痛,口干少津'
text_cu = " "+text+" "



# text_cuy = text + " "
text_cuy = text
# print(len("代诉：春节前右侧胸胁疼痛,现住院化疗,但不能饮食,头昏,恶心欲吐"))
model_input = tokenizer([text])
input_ids = torch.LongTensor(model_input['input_ids'])
# tokens = tokenizer.convert_ids_to_tokens(model_input['input_ids'])
output = model(input_ids)
#

translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,num_beams=3,num_return_sequences=1)
print(output.past_key_values[0][0].shape)
print(output.past_key_values[0][0][0][1][1][1])
# translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,num_beams=8,num_return_sequences=5)
# translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,do_sample=True,num_beams=8,top_k=50, temperature=0.7,num_return_sequences=1,min_length=1)
# translation = model.generate(input_ids,return_dict_in_generate= True , output_scores= True,output_attentions=True,do_sample=True)
print("attentions")
print(type(translation))
print(translation.sequences)
print(translation.sequences_scores)
print(translation.beam_indices[0])
# print(translation.scores[1].shape)
print(len(translation.cross_attentions))
print(len(translation.cross_attentions[3]))

# print(translation.cross_attentions[3][11].shape)
# print(translation.cross_attentions[1][0][3][3].shape)
# print(translation.cross_attentions[1][0][0])
# attention = translation.encoder_attentions[10][0][1].detach().numpy()
# print(attention.reshape(16,68))
count = 0
df_all = []
for a in range(0,6):
    df = []
    for i in translation.cross_attentions:

        print(count)
        count += 1
        dfin = []
        count = 0
        for j in i[a][0][0][0]:
            if count == 0:
                dfin.append(0.08/4)
                count += 1
            else:
                data = j.numpy()
                # print(j)
                data = str(data)
                data = float(data)
                dfin.append(data)
        df.append(dfin)
    df = numpy.array(df)
    df_all.append(df)
    # for i in df:
    #     print(i)

# df = df_all[0]+df_all[1]+df_all[2]+df_all[3]+df_all[4]+df_all[5]
df = df_all[0]





result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
print(result)



import numpy as np

text = [i for i in " "+text+" "]
target = [i for i in " "+result[0].replace(" ",'')+" "]
tar = " "+result[0].replace(" ",'')+" "
ta = result[0].replace(" ",'').split(",")
print(ta)
dtf = []
dtt = []
count = 0
inn = [0 for i in range(0,len(i))]
for i,j in zip(df,tar):
    if j == " " or j == ",":
        if count == 0:
            inn = [0 for i in range(0, len(i))]
            count = 0
        else:
            inne = []
            for s in inn:
                inne.append(s/count)
            dtt.append(inne)
            inn = [0 for i in range(0, len(i))]
            count = 0
        # if count == 0:
        #     inn = [0 for i in range(0, len(i))]
        #     dtf.append(inn)
        #     count = 0
        # if count > 0:
        #     for e in range(1,count+1):
        #         dtf.append(inn)
        #     inn = [0 for i in range(0, len(i))]
        #     dtf.append(inn)
        #     count = 0

    # elif j == ",":
    #     if count>0:
    #         for e in range(0,count):
    #             dtf.append(inn)
    #     inn = [0 for i in range(0,len(i))]
    #     dtf.append(inn)
    #     count = 0
    else:
        count += 1
        for a in range(0,len(i)):
            inn[a] += i[a]

for i in dtt:
    print(i)




import jieba
# jieba.load_userdict('ciku.txt')
text_cut = jieba.cut(text_cu,cut_all=False)

text_cutall = [i for i in text_cut]
print('[=================')
print(text_cutall)
print(len(text_cu))
print(len(text_cutall))
print(len(dtt[0]))
dttt = []
count = 0
tt = []
for i in text_cutall:
    for j in i:
       tt.append(count)
    count += 1
print(text_cutall)
print(tt)

for a in dtt:
    dtin = []
    dict = {}
    dict_count = {}
    for x in range(0,len(text_cutall)):
        dict[x] = 0
        dict_count[x] = 0

    for i,j in zip(a,tt):
        dict[j] += i
        dict_count[j] += 1
    print(dict)
    print(dict_count)
    print('00==00')
    for e in dict:
        try:
            dict[e] = dict[e]/dict_count[e]
        except:
            pass
    # print(dict)
    count = 0
    for e in dict:
        if count == 0:
            dtin.append(dict[e])
        # count += 1
    print(len(dtin))
    print(len(tt))
    print(len(a))
    dttt.append(dtin)



text_cutT = jieba.cut(text_cuy,cut_all=False)

text_cutallT = [i for i in text_cutT]




dtts = []
for i in dttt:
    count = 0
    dttsin = []
    for j in i:
        if count != 0:
            dttsin.append(j)
        count += 1
    dtts.append(dttsin)



for i in dtts:
    del(i[-1])



harvest = np.array(dtts)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['STSong'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #解决负号“-”显示为方块的问题
plt.xticks(np.arange(len(text_cutallT)), labels=text_cutallT,
                     rotation=45, rotation_mode="anchor", ha="right",fontsize=20,weight='normal')
plt.yticks(np.arange(len(ta)), labels=ta,fontsize=20,weight='bold')
plt.imshow(harvest)
plt.tight_layout()
plt.show()










# for i in attention:
#
#     for j in i:
#         if j <= 0.2:
#             dfin.append(j)
#         else:
#             dfin.append(0)
#     df.append(dfin)
#     dfin = []









# with open('relitu.txt','w') as f:
#     for i in attention:
#         for j in i:
#             f.write(str(j)+'\t')
#         f.write('\n')
# df = []
# dfin = []
# for i in attention:
#
#     for j in i:
#         if j <= 0.2:
#             dfin.append(j)
#         else:
#             dfin.append(0)
#     df.append(dfin)
#     dfin = []
# import pandas as pd
# df = pd.DataFrame(df)#,index=list('abc'), columns=list('ABC')
# import seaborn as sns
# import matplotlib.pyplot as plt
# result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
# print(result)
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False   #解决负号“-”显示为方块的问题
# figure, ax = plt.subplots(figsize=(15,15))
# # sns.heatmap(yy.corr(), square=True, annot=True, ax=ax)
# sns.heatmap(df, square=False, annot=True, ax=ax)
# plt.savefig('1.png')
# result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
# print(result)
# print(len(translation.encoder_attentions))
# print(type(translation))
# print(translation.encoder_attentions[10].shape)
# print(translation.decoder_attentions[1][1].shape)
# print(translation.cross_attentions[1][1].shape)
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # sns.set()
# # uniform_data=pd.read_csv(r'oddata.csv')
# # uniform_data.index=uniform_data['站号']
# # uniform_data=uniform_data.drop(columns='站号')
# # print(uniform_data)
# # ax = sns.heatmap(uniform_data,vmin=0,vmax=400,center=50)
# # ax.set_title('地铁OD数据客流可视化')
# # ax.set_xlabel('上车站号')
# # ax.set_ylabel('下车站号')
# # #可显示中文
# # plt.rcParams['font.sans-serif']=['SimHei']
# # plt.rcParams['axes.unicode_minus'] = False
# # plt.show()
#
#
#
#
#
#
#
#
#
#





# print(translation.cross_attentions[1][11])
# # print(translation.decoder_attentions)
# # for i in translation.decoder_attentions:
# #     print(i)
# print(len(translation.decoder_attentions[15]))
# print(translation.decoder_attentions[15][0].shape)
# translation = model.generate(input_ids)
# print(translation.beam_indices[0][5])
# print(len(translation.encoder_attentions))
# print(len(translation.decoder_attentions))
# print(translation.encoder_attentions[11].shape)
# print(translation.decoder_attentions[1][11].shape)
# print(translation.encoder_attentions[11][0][1].shape)
# result = tokenizer.batch_decode(translation, skip_special_tokens=True)
# print(result)
# # print(output[0].detach().numpy()[0][6])
# # for i in out:
# #     print(i.shape)
# text ="代诉：肺癌脑部多发性转移,11月21日上海华山医院γ刀治疗,面色萎黄欠华,腰酸隐痛,两侧胁肋时有痛感,活动不利,食纳尚可,咳嗽,痰粘色黄"
# inputs = tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=500,
#             return_tensors="pt",
#         )
# input_ids = inputs.input_ids.to(model.device)
# print(input_ids)
# print(input_ids.shape)
# translation = model.generate(input_ids)
# result = tokenizer.batch_decode(translation, skip_special_tokens=True)
# print(result)
# print(model(input_ids).encoder_last_hidden_state)
# print(model(input_ids).encoder_last_hidden_state.detach().numpy())
# # print(model(input_ids).decoder_hidden_states)
# # print(model(input_ids).encoder_attentions)
# print(model(input_ids).cross_attentions)
from transformers import BartForConditionalGeneration, BartTokenizer

# model = BartForConditionalGeneration.from_pretrained("./output1", forced_bos_token_id=0)
# # tok = BartTokenizer.from_pretrained("facebook/bart-large")
# example_phrase = "代诉：肺癌脑部多发性转移,11月21日上海华山医院γ刀治疗,<mask>色萎黄欠华,腰酸隐痛,两侧胁肋时有痛感,活动不利,食纳尚可,咳嗽,痰粘色黄"
# batch = tokenizer(example_phrase, return_tensors="pt")
# generated_ids = model.generate(batch["input_ids"])
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
# from transformers import BartTokenizer, BartForConditionalGeneration

# # tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# model = BartForConditionalGeneration.from_pretrained("./output1")
#
# TXT = "代诉：肺癌脑部多发性转移,11月21日上海华山医院γ刀治疗,<mask>色萎黄欠华,腰酸隐痛,两侧胁肋时有痛感,活动不利,食纳尚可,咳嗽,痰粘色黄"
# input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
# logits = model(input_ids).logits
#
# masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
# probs = logits[0, masked_index].softmax(dim=0)
# values, predictions = probs.topk(5)
#
# print(tokenizer.decode(predictions).split())

from transformers import BartTokenizer, BartModel
import torch

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# model = BartModel.from_pretrained("./output1")
#
# inputs = tokenizer("代诉：肺癌脑部多发性转移,11月21日上海华山医院γ刀治疗,<mask>色萎黄欠华,腰酸隐痛,两侧胁肋时有痛感,活动不利,食纳尚可,咳嗽,痰粘色黄", return_tensors="pt").input_ids.to(model.device)
# outputs = model(inputs)
#
# last_hidden_states = outputs.last_hidden_state
#
# print(last_hidden_states.shape)

#
# import pandas as pd
# df = pd.DataFrame(df)#,index=list('abc'), columns=list('ABC')
# import seaborn as sns
# import matplotlib.pyplot as plt
# result = tokenizer.batch_decode(translation.sequences, skip_special_tokens=True)
# print(result)
# print(len(" "+result[0].replace(' ',"")+" "))
# print(df.shape)
# print(len(" 代诉：春节前右侧胸胁疼痛,现住院化疗,但不能饮食,头昏,恶心欲吐 "),len(df[12]))
# xticks = [i for i in " "+result[0].replace(' ',"")+" "]
# yticks = [i for i in " 代诉：春节前右侧胸胁疼痛,现住院化疗,但不能饮食,头昏,恶心欲吐 "]
#
#
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False   #解决负号“-”显示为方块的问题
# figure, ax = plt.subplots()
# # sns.heatmap(yy.corr(), square=True, annot=True, ax=ax)
# sns.heatmap(df, square=False, annot=False, ax=ax)
# plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
# plt.xticks(plt.xticks()[0], labels=xticks)
# plt.savefig('1.png')
