# Human review instructions / 검수 안내

Reviewer code / 검수자 코드: **EN-R1-SSJKF**   Language / 언어: **English**   Items / 문항: **200**

Your code identifies you in this study instead of your name. Keep it.
이 코드는 실명 대신 사용하는 식별자입니다.

## 0. Independence / 독립성

The other reviewer of your language received the same 200 sentence pairs, with different review IDs and in a
different order. The random IDs only avoid exposing the answer key; they do not make the files incomparable.
Independence therefore depends on you: do not look at the other reviewer's file or at any answer key, do not discuss
any item before both of you have submitted, and submit your own judgments only.
같은 언어의 다른 검수자도 같은 200문항을 받았습니다(검수 ID와 순서만 다름). 무작위 ID는 정답 노출을 줄이는
장치일 뿐 두 파일을 비교할 수 없게 만드는 것은 아닙니다. 따라서 독립성은 절차로 확보합니다: 상대 파일이나 정답 자료를
보지 말고, 두 사람이 모두 제출하기 전에는 문항을 상의하지 말고, 본인 판정만 제출하십시오.

## 1. Item judgments / 문항 판정 (`items-EN-R1-SSJKF.csv`)

Each row shows two sentences in English. Both may be wrapped in the same frame (a question, reported speech, or a
conditional); judge the statements inside the frame.
각 행에는 두 문장이 있습니다. 두 문장이 같은 틀(의문문·인용·조건문)에 들어 있을 수 있으며, 틀 안의 진술을 판정합니다.

- `judged_label` = **1** if the two embedded statements express the same content, **0** if they differ.
  For the meaning to be the same, B must follow from A **and** A must follow from B. One-directional entailment is 0.
  두 문장의 내부 진술이 같은 내용을 표현하면 **1**, 다르면 **0**으로 판정합니다. 의미가 같으려면 A에서 B뿐 아니라
  B에서 A도 성립해야 합니다. 한 방향만 성립하면 0입니다.
- `fluent` = **yes** if both sentences are grammatical and natural for a native speaker; **no** otherwise.
  두 문장이 모두 문법적이고 자연스러우면 **yes**, 아니면 **no**. `no`이면 `comment`에 이유를 적어 주십시오.
- `comment`: optional otherwise. 그 외에는 선택.

Task column / 과제 열 (what "same content" means for each task / 과제별 '같은 내용'의 뜻):
- `roles`: the same participants play the same roles in both clauses. Active/passive or word-order changes with the
  same role assignment are the same content. 두 절에서 같은 참여자가 같은 역할. 태·어순만 다르면 같은 내용.
- `negation`: the same statement is affirmed and the same statement is denied. 같은 진술이 긍정되고 같은 진술이 부정됨.
- `space`: the same relative positions. Mirror phrasings (A left of B / B right of A) are the same content. 같은 상대 위치.
- `quantity`: the same number of each kind. Listing order does not matter. 각 종류의 개체 수가 같음.

Rules / 규칙: fill only `judged_label`, `fluent`, `comment`. Do not edit `review_id`, the sentences, or `reviewer`;
do not delete or reorder rows. Save as CSV (UTF-8). In Excel choose "CSV UTF-8".
`judged_label`, `fluent`, `comment`만 채우십시오. 다른 열·행 순서를 바꾸지 마십시오. UTF-8 CSV로 저장하십시오.

## 2. Templates and word forms / 문장 틀·활용표

- `noun-forms-en.csv`: every noun phrase surface form (case, number). Mark errors by row.
  모든 명사구 표면형(격·수). 오류가 있는 행을 표시.
- `construction-examples-en.csv`: one example per construction. Check grammaticality and naturalness.
  구문별 대표 문장. 문법성·자연스러움 확인.
- `template-inventory.json`: vocabulary tables. 어휘 표.
- `template-checklist-EN-R1-SSJKF.csv`: record `checked` (yes/no), `issue` (yes/no) and comments per row.

## 3. Return / 제출

Return `items-EN-R1-SSJKF.csv` and `template-checklist-EN-R1-SSJKF.csv` to the operator. Keep a copy.
두 파일을 운영자에게 보내고 사본을 보관하십시오.
