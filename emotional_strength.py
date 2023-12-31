import pandas as pd
import openai
import numpy as np
import re
import tiktoken
import collections
import os

def prompt_formulator(m_comments):
    indexed_comment_string = "comment's index | comment\n----|----------\n" + "\n".join(
        [
            f"{i} | '" + m_comments.loc[i, 'comment'] + "'"
            for i in m_comments.index
        ]
    )
    prompt = f'Here is a list of buyer\'s comments on an e-cigarette product:\n\"{indexed_comment_string}\"' \
             "\nIn the exact order, rate each comment's emotional strength. Rate from 1 to 5, where 1 is the weakest " \
             "and 5 is the strongest. " \
             "If the comment is negative about the product, rate it with negative number -1 to -5, " \
             "where -1 is the weakest negative and -5 is the strongest negative. "\
             "\nYour response should consist of lines in the format as such: " \
             "[comment's index] : [comment's emotion strength rate]"
    return prompt

def get_response_from_gpt(m_comments):
    # comment: a string of a single comment
    # questions: a list of questions about the comment to be fed to GPT

    # sys_message = f'''You evaluate this comment "{m_comment}" against a list of questions.
    # It is very important that you answer the questions purely based on the content of the comment.
    # Don't make any interpretation beyond the exact words in the comment. Answer in yes and no only.
    # '''

    # sys_message = f'''You evaluate this consumer's comment about a e-cigarette product against a list of
    # {len(m_questions)} criteria. The comment is "{m_comment}". You take each criterion from the list
    # and answer a question that take a form like this: Does this comment mention [the criterion]?
    # I need {len(m_questions)} answers in a list indexed by numbers in the exact same order as the criteria list
    # and separated by one new line character.
    # You should give me one and only one answer for each criterion! Give me the answers in yes/no only.
    # Don't give me the questions.'''

    prompt = prompt_formulator(m_comments)

    print("prompt sent to gpt")
    return openai.ChatCompletion.create(
        messages=[
            {"role": "system",
             "content": "You evaluate the emotional strength of comments on an e-cigarette product against a list of criteria I provide you."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4",
        temperature=0,
        request_timeout=300,
    )

def divider(df, size):
    ref_arr = np.array(range(len(df)), dtype='i') // size
    return df.groupby(by=ref_arr)

comments = pd.read_csv("stiiizycomments.csv", index_col=0).dropna()
comments = comments.reset_index(drop=True).loc[:, :]
grouped_comments = divider(comments, 65)


tokeniser = tiktoken.encoding_for_model("gpt-4")
openai.api_key = os.getenv("api_key")

for key, group in grouped_comments:
    groupcp = group.copy()
    response = get_response_from_gpt(groupcp)
    num_comments = len(groupcp)
    list_of_resp = (response["choices"][0]["message"]["content"]).strip("\n").split("\n", num_comments - 1)
    print(response)
    for resp in list_of_resp:
        # resp: [index]:[emo str]
        a = resp.split(':', 1)
        try:
            index = int(re.findall("\d+", a[0])[0])
            strength = int(re.findall("\d+", a[1])[0])
            groupcp.loc[index, "strength"] = strength
            comments.loc[index, "strength"] = strength
        except IndexError:
            continue
    groupcp.to_csv("stiiizycomments_withstrength23.csv", mode='a', header=False)

comments.to_csv("stiiizycomments_withstrength_total23.csv")
