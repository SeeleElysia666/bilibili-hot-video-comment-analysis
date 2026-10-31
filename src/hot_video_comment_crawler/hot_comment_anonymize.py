# 导包
import hashlib
import pandas as pd
import re

# 1.构造假名映射：按首次出现顺序编号，保证同一用户在不同文件里编号一致
def build_mapping(series):
    mapping = {}
    for value in series:
        if pd.isna(value) or value in mapping:
            continue
        mapping[value] = len(mapping) + 1
    return mapping

# 2.评论id加盐哈希：真实评论id可反查到具体用户，替换成不可逆的哈希串
salt = 'bilibili-hot-video-comment-analysis_2026'

def hash_comment_id(comment_id):
    return 'c_' + hashlib.sha256((salt + str(comment_id)).encode('utf-8')).hexdigest()[:12]

# 2.5正文脱敏规则：只替换身份列还不够，评论正文里也会出现 @他人昵称 和带个人追踪参数的链接
# 带协议的完整链接
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
# 不带 http:// 前缀的裸域名（b23.tv/xxx、pan.baidu.com/share/link?shareid=xxx），分享参数同样能反查到分享者
domain_pattern = re.compile(
    r'(?:[a-zA-Z0-9][a-zA-Z0-9\-]*\.)+(?:com|cn|net|org|tv|me|io|cc|info|biz|top|xyz|link)(?:/[^\s，。；、）]*)?')
# B站昵称里允许出现 - . _ 等符号（例如 @-LKs-），只匹配 \w 和汉字会漏掉这类账号
mention_pattern = re.compile(r'@[\w\u4e00-\u9fa5.\-·]{1,30}')
# 自检用：掩码后的提及是「@ 加等长 *」，正常不会再命中这个特征
unmasked_mention_pattern = re.compile(r'@[A-Za-z0-9_\u4e00-\u9fa5.\-·]')

# 2.6用等长占位符覆盖：只换内容不改变长度，下游的过滤规则与长度统计因此完全不受影响
def mask_text(text):
    if not isinstance(text, str):
        return text
    # 先处理 URL，避免链接内部的 @ 被误判成提及
    text = url_pattern.sub(lambda m: '*' * len(m.group()), text)
    text = domain_pattern.sub(lambda m: '*' * len(m.group()), text)
    text = mention_pattern.sub(lambda m: '@' + '*' * (len(m.group()) - 1), text)
    return text

# 3.按映射表替换三列身份字段，并对正文做脱敏
def anonymize(comments, user_name_map, user_id_map):
    comments['用户名'] = comments['用户名'].map(lambda s: user_name_map.get(s)).map(lambda i: f'用户{i:04d}' if i else s)
    comments['用户id'] = comments['用户id'].map(lambda s: user_id_map.get(s)).map(lambda i: f'uid_{i:04d}' if i else s)
    comments['评论id'] = comments['评论id'].apply(hash_comment_id)
    comments['评论内容'] = comments['评论内容'].apply(mask_text)
    return comments

# 4.脱敏原始评论数据
def hot_comment_anonymize():
    # 4.1读取原始数据（不动原文件，只读）
    comments = pd.read_csv('../../data/hot_comments.csv')
    comments_clean = pd.read_csv('../../data/hot_comments_clean.csv')

    # 4.2映射表以原始数据为准，清洗数据共用同一张表
    user_name_map = build_mapping(comments['用户名'])
    user_id_map = build_mapping(comments['用户id'])
    print(f'用户名 {len(user_name_map)} 个，用户id {len(user_id_map)} 个')

    # 4.3替换身份字段并保存
    comments = anonymize(comments, user_name_map, user_id_map)
    comments_clean = anonymize(comments_clean, user_name_map, user_id_map)
    comments.to_csv('../../data/hot_comments_anonymized.csv', index=False, encoding='utf-8-sig')
    comments_clean.to_csv('../../data/hot_comments_clean_anonymized.csv', index=False, encoding='utf-8-sig')
    print(f'保存文件成功：{comments.shape[0]} 条原始评论，{comments_clean.shape[0]} 条清洗评论')

    # 4.4自检：身份列是否还残留原始值，评论id是否已哈希且唯一，正文是否还有 @昵称 / 链接
    for name, frame in [('原始评论', comments), ('清洗评论', comments_clean)]:
        name_left = set(frame['用户名']) & set(user_name_map)
        id_left = set(frame['用户id']) & set(str(k) for k in user_id_map)
        # 占位符是 @**** 与等长的 *，都匹配不上这些正则，所以这里数出来的就是真正的残留
        mention_left = sum(len(unmasked_mention_pattern.findall(text)) for text in frame['评论内容'])
        url_left = sum(len(url_pattern.findall(text)) + len(domain_pattern.findall(text)) for text in frame['评论内容'])
        print(f'{name}：用户名残留 {len(name_left)} 个，用户id残留 {len(id_left)} 个，'
              f'评论id 唯一 {frame["评论id"].nunique()} 个，全部已哈希 {frame["评论id"].str.startswith("c_").all()}，'
              f'正文残留 @提及 {mention_left} 处，残留链接 {url_left} 处')

# 5.测试
if __name__ == '__main__':
    hot_comment_anonymize()
