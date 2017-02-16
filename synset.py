synset= [
        "jisu 1",
        "mijoo 2",
        "kei 3"
]
synset_map={}

for i, l in enumerate(synset):
    label, desc = l.split(' ', 1)
    synset_map[label] = {"index": i, "desc": desc}