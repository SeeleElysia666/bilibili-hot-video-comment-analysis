# 导包
import requests
import time
import csv
import hashlib # 哈希模块
import urllib.parse

# 1.配置
# 1.1B站入站必刷视频接口地址
url = 'https://api.bilibili.com/x/web-interface/popular/precious'

# 1.2模拟浏览器
headers = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0',
    'Referer': 'https://www.bilibili.com/v/popular/history',
    'Cookie' : "你的cookie"
}

# 1.3接口请求参数
params = {
'page_size' : '100',
'page' : '1',
'web_location' : '333.934',
'w_rid' : '29525fc6ba375cdaad6ff42253a9da7b',
'wts' : '1784715389'
}

# 2.爬取视频信息
def crawler_hot_videos():
    # 2.1请求入站必刷视频接口
    response1 = requests.get(url, headers=headers, params=params)

    # 2.2提取视频列表数据
    data_list = response1.json()['data']['list']

    # 2.3存储视频信息的列表
    video_data = []

    # 2.4遍历视频列表，提取视频元信息
    for item in data_list:
        oid = item['aid']  # 视频aid
        bvid = item['bvid']  # 视频bv号
        title = item['title']  # 视频标题
        tname = item['tname']  # 视频分区
        up_name = item['owner']['name']  # UP主名称
        view = item['stat']['view']  # 播放量
        like = item['stat']['like']  # 点赞数
        coin = item['stat']['coin']  # 硬币数
        favorite = item['stat']['favorite']  # 收藏数
        share = item['stat']['share']  # 分享数
        reply_total = item['stat']['reply']  # 评论总数
        achievement = item['achievement']  # 视频说明

        video_data.append([oid, bvid, title, tname, up_name, view, like, coin, favorite, share, reply_total, achievement])
        time.sleep(0.5)  # 增加延时

    # 2.5保存视频原始数据到
    with open('../../data/hot_video.csv', 'w', newline='', encoding='utf-8-sig') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(
            ['aid', 'bvid', '标题', '分区', 'up主', '播放', '点赞', '硬币', '收藏', '分享', '评论数', '说明'])
        writer1.writerows(video_data)

    print('视频信息爬取完毕')
    return video_data


# 3.爬评论信息
def crawler_video_comments(video_data):
    # 3.1存储所有评论
    all_comments = []

    # 3.2B站wbi签名盐值，用于MD5生成w_rid
    md5_str = 'ea1db124af3c7062474693fa704f4ff8'

    # 3.3遍历每一条视频，抓取评论
    for video in video_data:
        aid = video[0]
        bvid = video[1]
        print('正在爬：', bvid)

        count = 0  # 当前视频已抓取评论计数
        offset = ''  # 评论分页游标，用于下一页
        page_num = 0  # 评论页码

        # 3.3循环抓取，抓100条评论
        while count < 100:
            wts = str(int(time.time()))  # 当前时间戳，wbi签名参数

            # 3.4评论接口固定参数
            mode = 3 # 为2时爬取的是最新评论，为3时爬取的是热门评论
            oid = aid
            plat = 1
            type_ = 1
            web_location = '1315875'

            # 3.5分页游标json字符串
            pagination_str = '{"offset":"' + offset + '"}'

            # 3.6WBI签名：拼接参数字符串 + 盐值，MD5加密得到w_rid
            if page_num == 0:
                code = f"mode={mode}&oid={oid}&pagination_str={urllib.parse.quote(pagination_str)}&plat={plat}&seek_rpid=&type={type_}&web_location={web_location}&wts={wts}" + md5_str
            else:
                code = f"mode={mode}&oid={oid}&pagination_str={urllib.parse.quote(pagination_str)}&plat={plat}&type={type_}&web_location={web_location}&wts={wts}" + md5_str

            MD5 = hashlib.md5()
            MD5.update(code.encode('utf-8'))
            w_rid = MD5.hexdigest()

            # 3.7url编码分页参数，拼接完整评论接口url
            p = urllib.parse.quote(pagination_str, safe=':')
            if page_num == 0:
                video_url = f"https://api.bilibili.com/x/v2/reply/wbi/main?oid={oid}&type={type_}&mode={mode}&pagination_str={p}&plat=1&seek_rpid=&web_location=1315875&w_rid={w_rid}&wts={wts}"
            else:
                video_url = f"https://api.bilibili.com/x/v2/reply/wbi/main?oid={oid}&type={type_}&mode={mode}&pagination_str={p}&plat=1&web_location=1315875&w_rid={w_rid}&wts={wts}"

            # 3.8评论请求headers，Referer绑定当前视频bv号
            video_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0',
                'Referer': 'https://www.bilibili.com/video/' + bvid,
                'Cookie': "你的cookie"
            }

            # 3.9请求评论接口
            response2 = requests.get(video_url, headers=video_headers)
            data_video = response2.json()

            # 3.10接口无data字段：UP关闭评论，跳出当前视频循环
            '''
            实际爬取中发现有一个视频关闭了评论区
            '''
            if 'data' not in data_video:
                print(f'{bvid}UP主已关闭评论区')
                break

            # 3.11获取当前页评论列表
            replies = data_video['data']['replies']

            # 3.12遍历单页评论，提取字段
            for item in replies:
                if count >= 100:
                    break
                name = item['member']['uname']  # 评论用户名
                uid = item['member']['mid']  # 用户uid
                level = item['member']['level_info']['current_level']  # 用户等级
                rpid = item['rpid']  # 评论id
                text = item['content']['message']  # 评论文本
                likes = item['like']  # 评论点赞数
                all_comments.append([bvid, name, uid, level, rpid, text, likes])
                count = count + 1

            # 3.13更新分页游标，获取下一页
            try:
                offset = data_video['data']['cursor']['pagination_reply']['next_offset']
            except:
                offset = ''
            page_num = page_num + 1

            time.sleep(1)  # 评论页延时

        print(f'当前视频成功爬取{count}条')
        time.sleep(1.5)  # 视频延时

    # 3.14保存评论原始数据
    with open('../../data/hot_comments.csv', 'w', newline='', encoding='utf-8-sig') as f2:
        writer2 = csv.writer(f2)
        writer2.writerow(['bvid', '用户名', '用户id', '等级', '评论id', '评论内容', '点赞数'])
        writer2.writerows(all_comments)

    print('评论信息爬取完毕')
    return all_comments

# 4.测试
if __name__ == '__main__':
    video_result = crawler_hot_videos()
    comment_result = crawler_video_comments(video_result)