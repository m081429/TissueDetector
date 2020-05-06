import operator
import random
train=0.7
test=0.15
val=0.15

file_reader = open("tmp.xls")
lista=[]
num_cat={}
for i in file_reader:
    i = i.strip()
    arr = i.split("\t")
    if arr[1] in num_cat:
        num_cat[arr[1]]+=1
    else:
        num_cat[arr[1]]=0
    arr[2] = int(arr[2])
    lista.append(arr)
file_reader.close()
listb = sorted(lista, key=operator.itemgetter(1, 2))

low_num=0
for w in sorted(num_cat, key=num_cat.get, reverse=True):
    #print(w, num_cat[w])
    low_num=num_cat[w]

train_num = int(low_num*train)
val_num = int(low_num*val)
test_num = low_num-train_num-val_num

#print(train_num,test_num,val_num)
num_cat_perc={}
for key in num_cat:
    train_perc=float(train_num/num_cat[key])
    val_perc=float(val_num/num_cat[key])+train_perc
    test_perc=float(test_num/num_cat[key])+val_perc 
    num_cat_perc[key] = [train_perc,val_perc,test_perc]
    #print(key,[train_perc,val_perc,test_perc])

for i in listb:
    name=i[0]
    cat = i[1]
    num = i[2]
    status=""
    flt = random.random()
    if flt<num_cat_perc[cat][0]:
        status="train"
    elif  flt<num_cat_perc[cat][1]:
        status="val"
    elif  flt<num_cat_perc[cat][2]:
        status="test"
    else:
        status="extra"
    print(name+"\t"+cat+"\t"+str(num)+"\t"+status)