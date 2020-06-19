import json

f1 = open("/media/janzz11/Backup_Drive/rs_3_fetuared_embedded_flair_bpemb_train.jsonl", "r")

#f2 = open("/media/janzz11/Backup_Drive/rs_3_classes.jsonl", "w")

classes = {}

classes_count = {}

c = 0

for line in f1:

    x = json.loads(line)

    print(len(x['encoded_entity']))

    for elem in x["labels"]:

        if elem not in classes:

            classes[elem] = c

            classes_count[elem] = 1

            print(elem, c)

            c+=1
        else:
            classes_count[elem]+=1


print()

print(classes)
print()

print(classes_count)

json.dump(classes,f2, ensure_ascii=False)