# 导包
import re
import pandas as pd

# 0.复用的正则，与脱敏脚本 hot_comment_anonymize.py 保持一致
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
domain_pattern = re.compile(
    r'(?:[a-zA-Z0-9][a-zA-Z0-9\-]*\.)+(?:com|cn|net|org|tv|me|io|cc|info|biz|top|xyz|link)(?:/[^\s，。；、）]*)?')
mention_pattern = re.compile(r'@[\w\u4e00-\u9fa5.\-·]+')

# 1.正则化处理清洗后的评论
def re_processed_comments():
    comments = pd.read_csv('../../data/hot_comments_clean.csv')

    # 1.1去除链接：带协议的完整链接 + 不带前缀的裸域名
    comments['评论内容'] = comments['评论内容'].apply(lambda s: url_pattern.sub('', s))
    comments['评论内容'] = comments['评论内容'].apply(lambda s: domain_pattern.sub('', s))

    # 1.2去除@用户名：B站昵称里可能带 - . _ 等符号
    comments['评论内容'] = comments['评论内容'].apply(lambda s: mention_pattern.sub('', s))

    # 1.3去除B站表情 [xxx]
    comments['评论内容'] = comments['评论内容'].apply(lambda s: re.sub(r'\[.+?\]', '', s))

    # 1.4去除emoji
    emoji = re.compile("["
                        u"\U0001F600-\U0001F64F"  # 表情符号
                        u"\U0001F300-\U0001F5FF"  # 符号图画
                        u"\U0001F680-\U0001F6FF"  # 交通地图符号
                        u"\U0001F1E0-\U0001F1FF"  # 旗帜
                        u"\U00002700-\U000027BF"  # Dingbats
                        u"\U00002600-\U000026FF"  # 杂符号
                        u"\U0001F900-\U0001F9FF"  # 新增补表情
                           "]+", flags=re.UNICODE)
    comments['评论内容'] = comments['评论内容'].apply(lambda s: emoji.sub('', s))

    # 1.5多空白压缩单空格
    comments['评论内容'] = comments['评论内容'].apply(lambda s: re.sub(r'\s+', ' ', s).strip())

    # 1.6过滤清洗后空字符串
    comments = comments[comments['评论内容'] != ''].copy()

    # 1.7保存文件
    processed_comments = comments[['bvid', '评论内容', '点赞数']]
    processed_comments = processed_comments.dropna()
    processed_comments.to_csv('../../data/hot_comments_clean_processed.csv', index=False, encoding='utf-8-sig')
    print('保存文件成功')

# 2.测试
if __name__ == '__main__':
    re_processed_comments()
