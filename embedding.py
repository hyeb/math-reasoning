from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np
import tqdm



EMBED_MODEL_NAME = "dragonkue/bge-m3-ko" #https://huggingface.co/dragonkue/BGE-m3-ko
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
FILE_PATH = "/"


file_path = ""
data = pd.read_csv(file_path)

questions = data['question']

embeddings = embed_model.encode(
    questions,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

data['embeddings'] = embeddings.tolist()
emb_matix = np.vstack(embeddings).astype("float32")

N, D = emb_matix.shape

norms = np.linalg.norm(emb_matix, axis=1, keepdims=True) + 1e-12
emb_norm = emb_matix / norms

HARD_THRESHOLD = 0.90
SOFT_THRESHOLD = 0.8

group_ids = np.full(N, -1, dtype=int)
current_group = 0

for i in tqdm.tqdm(range(N)):
    if group_ids[i] != -1: #이미 그룹에 속해있으면 스킵
        continue
    
    group_ids[i] = current_group #아직 그룹이 없는 질문이라면 새롭게 번호 부여
    
    #i이후 샘플들과 유사도 계산
    if i + 1 < N:
        sims = emb_norm[i].dot(emb_norm[i+1:].T) #(i,j) 쌍에서 항상 j>i인것만 비교

        #threshold 이상 인덱스들
        dup_rel_indices = np.where(sims >= HARD_THRESHOLD)[0] #위치 추출
        dup_index = dup_rel_indices + (i+1) #질문 index

        for j in dup_index:
            if group_ids[j] == -1:
                #i와 같은 그룹으로 묶음. 즉, i번째 질문과 유사도 이상인 모든 질문이 같으 그룹 번호를 가짐
                group_ids[j] = current_group

    current_group += 1 


df = data.reset_index().rename(columns={"index": "orig_index"})
df["duplicate_group_id"] = group_ids

#그룹별 대표 샘플 선택
rep_idx_per_group = (df.groupby("duplicate_group_id")["orig_index"].min().to_dict())

def is_representative(orig_index, group_id): #
    return rep_idx_per_group.get(group_id, -1) == orig_index

df["is_representative"] = df.apply(
    lambda r: is_representative(r["orig_index"], r["duplicate_group_id"]),
    axis=1
)

df["is_duplicated"] = ~df["is_representative"]

df_dedup = df.query("is_representative == True").copy()

print("원본 개수:", len(df))
print("중복 제거 후 개수: ", len(df_dedup))


df.to_csv(FILE_PATH + "math_qa_with_dup_info.csv", index=False)
df_dedup.to_csv(FILE_PATH + "math_qa_dedup.csv", index=False)