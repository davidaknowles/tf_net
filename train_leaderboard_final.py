train_test={}

for l in open("train_leaderboard_final.txt"):
    ss=l.strip().split("|")
    tf=ss[0].split("]")[0][1:]
    train_test[tf]={}
    filt=lambda g: [ s for s in g.replace("*","").split(", ") if len(s)>0 ]
    train_test[tf]["train"]=filt(ss[1])
    train_test[tf]["leaderboard"]=filt(ss[2])
    train_test[tf]["final"]=filt(ss[3])
    
to_test={ k:v["final"] for k,v in train_test.iteritems() if len(v["final"])>0 }

