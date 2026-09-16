"""Transparent baselines use sentence text only, never the generator's scene/label."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import re
import numpy as np
from scipy import optimize, sparse, special

from .data import noun_phrase, VERBS, load_rows
from .protocol import LANGUAGES, TASKS, digest, file_hash, write_json


@lru_cache(None)
def entity_matcher(lang):
    mapping={}
    for entity in product(range(12),range(6),range(4)):
        for case in ("nom","acc","dat"):
            mapping[noun_phrase(entity,lang,case).casefold()]=entity
    pattern=re.compile("|".join(re.escape(s) for s in sorted(mapping,key=len,reverse=True)),re.IGNORECASE)
    return pattern,mapping


@lru_cache(None)
def verb_matcher(lang):
    variants={}
    for i,v in enumerate(VERBS):
        words=v[:3] if lang=="en" else v[3:5] if lang=="de" else (v[5],("보지","반기지","뒤따르지")[i])
        variants.update({word.casefold():i for word in words})
    pattern="|".join(re.escape(s) for s in sorted(variants,key=len,reverse=True))
    if lang!="ko": pattern=r"\b(?:"+pattern+r")\b"
    return re.compile(pattern,re.IGNORECASE),variants


def roles_parse(text,lang):
    text=text.casefold()
    pattern,mapping=entity_matcher(lang)
    entities=[(mapping[m.group().casefold()],m.end()) for m in pattern.finditer(text)]
    verbs,lookup=verb_matcher(lang)
    actions=[lookup[m.group().casefold()] for m in verbs.finditer(text)]
    if len(entities)!=4 or len(actions)!=2:
        return None
    passive=(" by " in text.casefold()) if lang=="en" else (" wird " in text.casefold() or " wird" in text.casefold())
    result=[]
    for k in range(2):
        a,end_a=entities[2*k]; b,end_b=entities[2*k+1]
        if lang=="ko":
            if not text[end_a:].startswith(("이","가")): a,b=b,a
        elif passive:
            a,b=b,a
        result.append((a,actions[k],b))
    return tuple(sorted(result))


def hypothesis_features(rows,lang,dim=2048):
    ri,ci,values=[],[],[]
    verbs,lookup=verb_matcher(lang)
    for i,row in enumerate(rows):
        text=row["sentence_b"].casefold()
        tokens=re.findall(r"\w+",text)
        features=["word:"+word for word in tokens]
        sequence=[lookup[m.group().casefold()] for m in verbs.finditer(text)]
        features.append("verb-order:"+str(sequence))
        for f in features:
            ri.append(i); ci.append(int(digest(f)[:8],16)%dim); values.append(1.)
        ri.append(i); ci.append(dim); values.append(1.)
    return sparse.csr_matrix((values,(ri,ci)),shape=(len(rows),dim+1))


def run_baselines(root):
    train,held=load_rows(root,"train"),load_rows(root,"dev_a")
    report={}
    for lang in LANGUAGES:
        for task in TASKS:
            tr=[r for r in train if r["language"]==lang and r["task"]==task]
            te=[r for r in held if r["language"]==lang and r["task"]==task]
            x,z=hypothesis_features(tr,lang),hypothesis_features(te,lang)
            y=np.array([r["label"] for r in tr])
            def fun(w):
                logit=x@w
                return float((np.logaddexp(0,logit)-y*logit).mean()+.001*np.square(w).sum()), np.asarray(x.T@(special.expit(logit)-y)).ravel()/len(y)+.002*w
            fit=optimize.minimize(fun,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":100})
            accuracy=float(np.mean((z@fit.x>=0)==np.array([r["label"] for r in te])))
            report[f"hypothesis_only/{lang}/{task}"]=dict(accuracy=accuracy,audit_required=accuracy>.55,
                optimizer_success=bool(fit.success),uses_sentence_a=False)
        subset=[r for r in held if r["language"]==lang and r["task"]=="roles"]
        correct,covered=0,0
        for row in subset:
            a,b=roles_parse(row["sentence_a"],lang),roles_parse(row["sentence_b"],lang)
            if a is not None and b is not None:
                covered+=1; correct+=int((a==b)==bool(row["label"]))
        report[f"grammar_cue/{lang}/roles"]=dict(coverage=covered/len(subset),accuracy=correct/covered if covered else None,
              audit_required=covered!=len(subset),uses_scene_metadata=False,
              interpretation="Legitimate order/case/passive and predicate binding; high accuracy is not an artifact failure.")
    write_json(Path(root)/"cue-baselines.json",report)
    write_json(Path(root)/"cue-baselines.meta.json",dict(data_files={split:file_hash(Path(root)/f"{split}.jsonl") for split in ("train","dev_a")},
               implementation_hash=file_hash(__file__)))
    return report
