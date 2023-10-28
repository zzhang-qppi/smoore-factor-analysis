def XuQiuManZu(count_data):
    # 需求满足程度
    # 1）所有负面或问题评价占比
    total_count = count_data["count"].sum()  # 所有标签提及数综合
    negative_count = count_data.loc[count_data.emotion < 0, "count"].sum()  # 负面标签提及数综合
    return negative_count / total_count


def XiYinLi(comment_emotion_data):
    # 吸引力
    # 所有负面评价的情感强度均值
    return comment_emotion_data[[comment_emotion_data.strength < 0], "strength"].mean()  # 消极评论的情感强度均值


def DuTeXing(count_data):
    # 独特性
    # 新口味，新功能讨论集中度
    total_count = count_data["count"].sum()  # 所有标签提及数综合
    new_flav_count = count_data.loc[count_data.label == "new flavors", "count"]
    new_func_count = count_data.loc[count_data.label == "new functions", "count"]  # “新口味” 和 “新功能” 提及数总和
    return (new_func_count+new_flav_count) / total_count


def GouMaiYiYuan(comment_emotion_data):
    # 购买与了解意愿
    # 2）整体好评率，情感分析积极评价占比
    n_comment = len(comment_emotion_data)  # 评论数目
    positive_count = len(comment_emotion_data(comment_emotion_data.emotion > 0))  # 积极评论数目
    return n_comment / positive_count
