"""Create an editable XMind map of this project's AIO2-inspired implementation."""
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import json
import time

OUT = Path(__file__).resolve().parent
C = 'urn:xmind:xmap:xmlns:content:2.0'
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', C)
ET.register_namespace('svg', SVG)
STAMP = str(int(time.time()*1000))
book = ET.Element(f'{{{C}}}xmap-content', {'version':'2.0','timestamp':STAMP})
counter = 0
def add(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent, f'{{{C}}}{tag}', attrs)
    e.text = text
    return e
def topic(parent, title, style='blue', note='', id=None, position=None):
    global counter
    counter += 1
    t = add(parent,'topic',id=id or f'topic_{counter}',timestamp=STAMP,
            **{'style-id':style,'structure-class':'org.xmind.ui.logic.right'})
    add(t,'title',title)
    if note:add(add(t,'notes'),'plain',note)
    if position:
        add(t,'position',**{f'{{{SVG}}}x':str(position[0]),f'{{{SVG}}}y':str(position[1])})
    return t
def kids(t, kind='attached'):
    return add(add(t,'children'),'topics',type=kind)
def sheet(id,title):
    s=add(book,'sheet',id=id,timestamp=STAMP)
    add(s,'title',title)
    return s
scope='本图依据本项目的 EMA 教师与在线临时补标代码；属于 AIO2 启发的任务适配，使用固定轮数触发，未完整复现原论文 ACT/O2C。默认参数为实验起点，不代表已验证最优配置。'
sections=[
('01 研究动机与方法范围','note',[
('目录存在漏标','USGS 未标注区域不等于已确认无断层。直接把潜在漏标区域作为背景，可能产生错误监督。'),
('教师提供候选，学生学习临时标签','利用 EMA 教师的预测筛选疑似漏标区域，保留原标注，并为本批次增加低权重正标签。'),
('项目实现：固定触发的保守在线补标',scope),
('新增候选需要核查','接纳区域仍可能为假阳性；不能据此声称发现真实断层或已经提升整体性能。')]),
('02 数据与学生模型','blue',[
('七通道地形输入','CTX 灰度地形、标准化水深、两个有效掩膜，以及 315°/75°/195° 三个方向的晕渲。教师和学生在该分支使用同一个训练裁剪。'),
('USGS 原始标签','矢量断层线 → 地理对齐 → 距离场 → distance < d0，形成 k0 监督标签。原始数据文件不被补标覆盖。'),
('卷积融合 → 三通道适配','head 提取局部特征，input_sam 卷积将特征映射为三通道表征。'),
('SAM-2 编码 → 多尺度融合解码','SAM-2 forward_image 提取多尺度特征，与局部跳跃特征融合后上采样。'),
('输出 k0 断层概率','网络输出 Softmax 多通道概率，k0Only 监督使用首个主任务通道；学生可训练参数通过梯度更新。')]),
('03 EMA 教师','green',[
('初始化：复制学生模型','教师与学生结构相同；初始化深拷贝学生参数。'),
('教师只推理，不反向传播','教师使用 eval/no_grad，requires_grad=False，不加入优化器。'),
('学生更新后，再更新教师','每次实际 optimizer.step() 完成后更新一次教师；梯度累积的每个微批次不会单独触发 EMA。'),
('EMA：θT ← αθT + (1−α)θS','θS 为刚更新的学生参数，θT 为历史教师参数。在线实验命令指定 α=0.98；通用 EMA 默认值为0.99，以实际配置为准。'),
('教师再用于下一批候选生成','对当前 batch，先用当前教师推理，再计算学生损失；EMA 更新影响后续 batch。')]),
('04 候选区域筛选','purple',[
('有效像素与有限预测','两个有效掩膜均为有效，输入与教师概率为有限数值。'),
('p ≥ 0.5：建立八连通区域','在完整教师预测上构建支持区域；中等概率连接带也参与连通性判断。'),
('区域内至少一个 p ≥ 0.9 的种子','不要求每一个候选像素均达到0.9；被接纳的是含高置信种子的整个支持连通区域。'),
('远离原标注及其3像素缓冲区','整个区域只要碰到原标注或缓冲区即整体拒绝，避免把原断层的外扩环当作新区域。'),
('区域至少16像素，面积不超过5%','单个候选区域面积 ≤ 当前裁剪有效面积的5%；所有候选总面积也需满足此限制。'),
('总面积超预算：本裁剪候选全拒绝','不通过裁掉部分像素凑面积上限。上述数值均为当前默认配置，可通过参数调整。')]),
('05 生成临时监督并计算损失','purple',[
('原始正标签保留','原标签与其他未接纳像素保持原样，不删除或永久改写原始正标签。'),
('接纳区域 → 本批次硬正标签','临时目标 = 原始标签 ∪ 接纳候选；每个 batch 用当前教师重新生成，候选可以出现或消失。'),
('候选权重 = λ × 本裁剪最大原空间权重','候选使用低权重，其他像素的空间权重不变；避免候选因远离原标注再次被距离权重削弱。'),
('同一个加权 Focal Tversky 损失','学生预测与临时目标比较；候选已替换其背景身份，不再同时作为背景产生假阳性惩罚。在线实验要求 PU=0。'),
('λ=0.2 并非占总损失20%','λ 是候选像素权重系数，不是损失占比或候选面积比例；无候选或λ=0时返回原标签和原权重。')]),
('06 训练循环与时间表','orange',[
('① 教师推理 → 筛选 → 临时标签','补标启用后，教师先预测，依据候选规则构造当前 batch 的临时目标与权重。'),
('② 学生预测 → 损失 → 反向传播','连续断层概率直接参与损失计算；梯度按配置累积、检查和裁剪。'),
('③ AdamW 更新学生 → EMA 更新教师','学生通过优化器学习，教师通过参数指数移动平均更新，随后进入下一批。'),
('第1–40轮：预热，不补标','使用原标签训练；教师仍随学生实际参数更新进行 EMA 更新。轮数采用从1开始的编号。'),
('第41–50轮：λ=0.05','按固定轮数启动在线补标。'),
('第51–80轮：λ从0.05线性升到0.2','第51轮0.055，第65轮0.125，第80轮0.2。'),
('第81轮起：λ=0.2，持续逐批补标','实际终止轮数由 epochs 指定；第80轮不是停止补标的时点。时间表可配置。')]),
('07 验证与汇报说明','note',[
('验证使用原始标签，不加入伪标签','固定中心裁剪、关闭随机增强，仅在有效像素上评估；PU=0。'),
('分别选择学生与教师的最佳模型','按各自固定验证损失选择，最佳轮数可以不同；测试集不参与轮次选择。'),
('补标计数不等于新增独立断层数','计数包含重复采样和增强裁剪，应结合实际 online 预览与人工核查。'),
('汇报一句话','在基础分割训练中，引入由学生参数滑动平均得到的教师模型，用其高置信预测筛选疑似漏标区域，并以较低权重临时加入监督，从而探索在不完整标注下训练断层分割模型。'),
('实现依据','aio2_ema.py、aio2_online.py、config.py 与 trainV10.py；本图未加入 UniMatch 双强视图或一致性损失分支。')])]

s=sheet('aio2_mindmap','01 AIO2 思维导图')
root=topic(s,'Earthquake-SAM + AIO2 启发方法\nEMA 教师与在线临时补标','title',scope,id='aio2_root')
branches=kids(root)
for heading,color,items in sections:
    b=topic(branches,heading,color)
    child=kids(b)
    for title,body in items:
        t=topic(child,title,color,body)
        t.set('branch','folded')
        topic(kids(t),body,'note')

s=sheet('aio2_flow','02 教师学生与补标流程')
root=topic(s,'AIO2 启发的训练循环\n本项目：固定触发的 EMA 教师 + 临时补标','title',scope,id='flow_root',position=(50,-380))
detached=kids(root,'detached')
nodes=[
('x','同一批七通道地形输入',-750,0,'blue','教师与学生使用同一训练裁剪。'),
('teacher','EMA 教师\n仅推理，无梯度',-420,-160,'green','先预测当前 batch；教师由学生初始化，并在实际学生更新后使用EMA更新。'),
('tp','教师断层概率',-90,-160,'green','使用 k0 概率构建候选支持区域。'),
('select','候选筛选\n置信度 · 连通性 · 距离 · 面积',260,-160,'purple','默认：p≥0.5 的八连通区域，至少一个p≥0.9种子；不碰原标签及3像素缓冲；至少16像素；单区及总面积≤有效面积5%。'),
('temporary','临时目标 + 像素权重\n原标签 ∪ 候选',650,-160,'purple','保留原正标签；候选使用λ×本裁剪最大原空间权重。每批重新生成，不写回原标签。'),
('original','USGS 原始监督标签',260,-430,'purple','矢量线 → 对齐 → 距离场 → d0阈值化。标签也用于候选排除区判断。'),
('student','学生分割模型\n融合 → SAM-2 → 解码',-420,150,'blue','学生参数通过梯度和AdamW更新。模型内部与基础版本一致。'),
('sp','学生断层概率',-90,150,'blue','使用连续预测计算损失。'),
('loss','加权 Focal Tversky\n临时监督 · PU=0',650,150,'orange','同一个损失计算；接纳候选替换原背景身份，不重复施加背景惩罚。'),
('update','反向传播 → AdamW\n更新学生参数',260,410,'orange','梯度可累积，实际 optimizer.step 后才更新教师。'),
('ema','EMA 更新教师\nθT ← αθT + (1−α)θS',-420,410,'green','在线实验命令使用α=0.98；以实际运行配置为准。更新后的教师用于后续批次。'),
('time','默认补标计划\n1–40轮：λ=0\n41–50轮：0.05\n51–80轮：升至0.2\n81轮起：保持0.2',1080,-160,'note','λ是候选像素权重系数，非总loss占比。固定时间表可配置。'),
('val','验证：原始标签\n无伪标签 · 无参数更新',1080,210,'note','固定中心裁剪；分别记录学生与教师损失并选择最佳模型。'),
]
for id,title,x,y,color,note in nodes:topic(detached,title,color,note,id=id,position=(x,y))
rels=add(s,'relationships')
edges=[
('x','teacher','输入','relation'),('x','student','输入','relation'),
('teacher','tp','','relation'),('tp','select','','relation'),
('original','select','排除已有标签附近区域','label_relation'),
('original','temporary','保留原标签','label_relation'),
('select','temporary','接纳候选','label_relation'),
('time','temporary','控制λ','label_relation'),
('temporary','loss','临时目标与权重','label_relation'),
('student','sp','','relation'),('sp','loss','预测值','relation'),
('loss','update','损失与梯度','feedback'),
('update','student','更新学生','feedback'),
('update','ema','实际step完成后','feedback'),
('ema','teacher','更新教师供后续批次使用','feedback')]
for i,(a,b,label,style) in enumerate(edges):
    r=add(rels,'relationship',id=f'aio2_rel_{i}',end1=a,end2=b,**{'style-id':style})
    if label:add(r,'title',label)

source=OUT/'Earthquake-SAM_模型与训练流程.xmind'
with zipfile.ZipFile(source) as z:
    styles=z.read('styles.xml')
    meta=z.read('meta.xml')
entries={'content.xml':ET.tostring(book,encoding='utf-8',xml_declaration=True),'styles.xml':styles,'meta.xml':meta}
manifest=ET.Element('manifest',{'xmlns':'urn:xmind:xmap:xmlns:manifest:1.0'})
for name in entries:ET.SubElement(manifest,'file-entry',{'full-path':name,'media-type':'text/xml'})
entries['META-INF/manifest.xml']=ET.tostring(manifest,encoding='utf-8',xml_declaration=True)
output=OUT/'Earthquake-SAM_AIO2方法_思维导图.xmind'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in entries.items():z.writestr(name,data)
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    for name in entries:ET.fromstring(z.read(name))
    parsed=ET.fromstring(z.read('content.xml'))
    ids=[e.get('id') for e in parsed.iter() if e.get('id')]
    assert len(ids)==len(set(ids))
    for sh in parsed.findall(f'{{{C}}}sheet'):
        valid={t.get('id') for t in sh.iter(f'{{{C}}}topic')}
        for r in sh.iter(f'{{{C}}}relationship'):
            assert r.get('end1') in valid and r.get('end2') in valid
    titles=[t.text for t in parsed.iter(f'{{{C}}}title')]
    assert any('未完整复现' in t for t in titles)
    assert any('PU=0' in t for t in titles)
    assert any('0.2' in t for t in titles)
markdown=['# Earthquake-SAM：AIO2 启发的 EMA 教师与在线临时补标','',scope,'']
for heading,_,items in sections:
    markdown.extend(['## '+heading,''])
    for title,body in items:markdown.extend(['- **'+title+'**：'+body])
    markdown.append('')
(OUT/'Earthquake-SAM_AIO2方法_导图说明.md').write_text('\n'.join(markdown),encoding='utf-8')
print(json.dumps({'file':output.name,'sheets':2,'topics':counter,'relationships':len(edges),'validation':'ZIP CRC, XML parsing, unique IDs, per-sheet relationship endpoints, key scope notes passed'},ensure_ascii=True))
