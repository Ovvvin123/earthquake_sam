from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import time
import json

OUT=Path(__file__).resolve().parent
C='urn:xmind:xmap:xmlns:content:2.0'
S='urn:xmind:xmap:xmlns:style:2.0'
SVG='http://www.w3.org/2000/svg'
FO='http://www.w3.org/1999/XSL/Format'
ET.register_namespace('',C)
ET.register_namespace('svg',SVG)
ET.register_namespace('fo',FO)
stamp=str(int(time.time()*1000))
def el(parent,tag,attrs=None,text=None,ns=C):
    e=ET.SubElement(parent,f'{{{ns}}}{tag}',attrs or {})
    if text is not None:e.text=text
    return e
book=ET.Element(f'{{{C}}}xmap-content',{'version':'2.0','timestamp':stamp})
def sheet(id,title):
    s=el(book,'sheet',{'id':id,'timestamp':stamp})
    el(s,'title',text=title)
    return s
def topic(parent,id,title,style='blue',note=None,pos=None):
    t=el(parent,'topic',{'id':id,'timestamp':stamp,'style-id':style,'structure-class':'org.xmind.ui.logic.right'})
    el(t,'title',text=title)
    if note:el(el(t,'notes'),'plain',text=note)
    if pos:el(t,'position',{f'{{{SVG}}}x':str(pos[0]),f'{{{SVG}}}y':str(pos[1])})
    return t
def children(t,kind='attached'):
    return el(el(t,'children'),'topics',{'type':kind})
def relationship(parent,id,start,end,title='',style='relation'):
    r=el(parent,'relationship',{'id':id,'end1':start,'end2':end,'style-id':style,'timestamp':stamp})
    if title:el(r,'title',text=title)

s1=sheet('sheet_flow','01 模型与训练流程')
root=topic(s1,'flow_title','Earthquake-SAM｜模型与训练循环','title',
           '自由主题与关系线均可编辑。图示为当前 mars_sam + k0Only 的基础监督流程；不展开可选 EMA、在线补标和 UniMatch。',(-50,-240))
floating=children(root,'detached')
nodes=[
('input','七通道输入\n地形×2 · 掩膜×2 · 晕渲×3',-810,0,'blue','顺序：CTX 灰度地形、标准化水深、CTX 有效掩膜、水深有效掩膜、315°/75°/195°晕渲。'),
('head','卷积特征融合\n融合七通道，提取局部特征',-530,0,'blue','head：卷积与残差块；对应 torchUnet.py conv_sam。'),
('adapter','三通道适配\n生成可学习的三通道表征',-240,0,'blue','input_sam 将局部特征映射为三通道。不是简单选择三个原始通道，也不表示真实 RGB 颜色。'),
('encoder','SAM-2 图像编码器\n提取多尺度特征',50,0,'blue','调用 sam.forward_image。当前运行检查确认图像编码器参数 requires_grad=True；训练代码使用较小的非零学习率。eval() 不等于冻结。'),
('decoder','特征融合与解码\n融合局部特征，逐级上采样',360,0,'blue','SAM 多尺度特征经卷积投影，与局部分支特征相加；upblock0/1/2 上采样，tail 输出。'),
('prediction','断层概率图\n取 k0 主任务通道',690,0,'green','网络完整输出为三个 Softmax 通道；k0Only 监督使用首个通道。损失使用连续概率，不先二值化预测。'),
('loss','计算损失\nFocal Tversky',690,250,'orange','比较连续预测与训练标签，结合像素权重。具体损失配置以训练命令为准。'),
('update','反向传播 → 参数更新\n计算梯度 → AdamW 更新',220,250,'orange','loss.backward() 计算梯度；按配置累积梯度、裁剪，opt.step() 更新参与前向计算且得到梯度的参数，随后清零梯度。'),
('catalog','USGS 断层目录\n已有断层矢量线',220,480,'purple','来源：项目中的 Qfaults_GIS/SHP/ca_offshore.shp。未标注区域不等于确认无断层。'),
('target','构建训练标签\n空间对齐 → 距离场 → 阈值化',690,480,'purple','以断层线生成距离场，训练时用 distance < d0 形成主任务标签。标签不属于七个输入通道。'),
('scope','阅读提示\n每批数据重复训练循环\n可先累积梯度，再更新参数',-580,340,'note','EMA 教师、在线临时补标和 UniMatch 属于后续扩展，可另加分支。')]
for id,title,x,y,style,note in nodes:topic(floating,id,title,style,note,(x,y))
rels=el(s1,'relationships')
for i,(a,b) in enumerate(zip(['input','head','adapter','encoder','decoder'],['head','adapter','encoder','decoder','prediction'])):
    relationship(rels,f'forward{i}',a,b)
relationship(rels,'skip','head','decoder','局部特征跳跃连接','skip')
relationship(rels,'pred_loss','prediction','loss','预测值')
relationship(rels,'label_loss','target','loss','目标值','label_relation')
relationship(rels,'catalog_target','catalog','target','生成标签','label_relation')
relationship(rels,'loss_update','loss','update','损失','feedback')
relationship(rels,'update_encoder','update','encoder','更新模型可训练参数','feedback')

s2=sheet('sheet_notes','02 模型说明与讲稿')
r2=topic(s2,'notes_root','Earthquake-SAM\n模型与训练说明','title')
c2=children(r2)
sections=[
('输入：七通道',[
('CTX 灰度地形','同一份 GEBCO 地形，经第2–98百分位拉伸并截断，随后映射到[-1,1]。'),
('标准化水深／高程','每块减去均值、除以标准差，保留未截断的相对高程变化；不再直接表示米。'),
('两个有效数据掩膜','分别标识 CTX 与水深是否有有效数据；1有效、0缺失。'),
('三个方向晕渲','方位角315°、75°、195°，突出不同走向的地形结构。')]),
('模型内部：特征提取与解码',[
('① 卷积融合','将七通道信息融合为局部特征。'),
('② 三通道适配','可学习卷积生成 SAM-2 输入表征。'),
('③ SAM-2 编码器','从预训练图像编码器获得多尺度特征。'),
('④ 特征融合与上采样','SAM 特征与局部跳跃特征融合，经自定义解码模块恢复空间分辨率。'),
('⑤ 输出断层概率','Softmax 输出；主任务使用 k0 通道的逐像素概率。')]),
('监督：USGS 标注',[
('矢量线 → 距离场 → 标签','将地理坐标与地形网格对齐，生成断层距离场，再按 d0 阈值形成训练目标。'),
('未标注不等于无断层','目录可能漏标；当前基础图只说明监督主流程。')]),
('训练循环',[
('① 前向预测','输入数据经过模型，得到断层概率图。'),
('② 计算损失','用 Focal Tversky 比较预测与标签，并结合像素权重。'),
('③ 反向传播','loss.backward() 计算各可训练参数的梯度。'),
('④ 更新参数','可累积多个批次梯度；裁剪后用 AdamW 执行 opt.step()，再清零梯度。'),
('⑤ 重复循环','对后续批次与训练轮次重复上述过程。')]),
('当前代码的关键区别',[
('SAM-2 图像编码器参与训练','当前参数状态为可训练，优化器学习率非零；旧说明“冻结主干”与当前实现不一致。'),
('连续概率直接用于损失','训练不需要先把预测阈值化为黑白图；阈值化主要用于输出展示或二值评估。'),
('验证和推理不更新参数','参数更新发生在训练阶段；验证用于监测和选模型。'),
('可选扩展另页介绍','EMA 教师、在线补标和 UniMatch 在基础流程上增加训练机制。')]),
('可直接使用的讲稿',[
('模型说明','首先融合七个输入通道并转换为三通道表征，随后利用 SAM-2 提取多尺度特征，再结合局部地形特征进行解码，输出逐像素的断层概率。'),
('训练说明','将预测与 USGS 目录生成的标签比较并计算损失，通过反向传播得到梯度，再由优化器更新模型参数，循环进行训练。')])]
for i,(title,items) in enumerate(sections):
    t=topic(c2,f'section{i}',title,['blue','blue','purple','orange','note','green'][i])
    ci=children(t)
    for j,(name,body) in enumerate(items):
        sub=topic(ci,f'detail{i}_{j}',name,note=body)
        topic(children(sub),f'body{i}_{j}',body,'note')

styles=ET.Element(f'{{{S}}}xmap-styles',{'version':'2.0'})
custom=el(styles,'styles',ns=S)
colors={'blue':('#edf6fb','#accbdf'),'green':('#e9f6ef','#abcdbb'),'orange':('#fff2e8','#e9bd99'),'purple':('#f1effa','#c5bfdf'),'note':('#f8fafc','#cbd5e1'),'title':('#19344b','#19344b')}
for id,(fill,line) in colors.items():
    st=el(custom,'style',{'id':id,'type':'topic'},ns=S)
    el(st,'topic-properties',{'shape-class':'org.xmind.topicShape.roundedRect','fill':fill,'line-color':line,'line-width':'1pt',f'{{{FO}}}font-family':'Microsoft YaHei',f'{{{FO}}}font-size':'20pt' if id=='title' else '14pt',f'{{{FO}}}color':'#ffffff' if id=='title' else '#19344b'},ns=S)
for id,color,shape in [('relation','#2476a7','org.xmind.relationshipShape.straight'),('label_relation','#8271a8','org.xmind.relationshipShape.straight'),('feedback','#cc6632','org.xmind.relationshipShape.curved'),('skip','#2476a7','org.xmind.relationshipShape.curved')]:
    st=el(custom,'style',{'id':id,'type':'relationship'},ns=S)
    el(st,'relationship-properties',{'shape-class':shape,'line-color':color,'line-width':'2pt','arrow-end-class':'org.xmind.arrowShape.triangle','arrow-begin-class':'org.xmind.arrowShape.none'},ns=S)
def xml(x):return ET.tostring(x,encoding='utf-8',xml_declaration=True)
meta=ET.Element('meta',{'xmlns':'urn:xmind:xmap:xmlns:meta:2.0'})
creator=ET.SubElement(meta,'Creator');ET.SubElement(creator,'Name').text='Earthquake-SAM diagram exporter'
ET.SubElement(creator,'Version').text='1.0'
manifest=ET.Element('manifest',{'xmlns':'urn:xmind:xmap:xmlns:manifest:1.0'})
entries={'content.xml':xml(book),'styles.xml':xml(styles),'meta.xml':xml(meta)}
for name in entries:ET.SubElement(manifest,'file-entry',{'full-path':name,'media-type':'text/xml'})
entries['META-INF/manifest.xml']=xml(manifest)
output=OUT/'Earthquake-SAM_模型与训练流程.xmind'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in entries.items():z.writestr(name,data)
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    for name in entries:ET.fromstring(z.read(name))
    parsed=ET.fromstring(z.read('content.xml'))
    ids=[e.get('id') for e in parsed.iter() if e.get('id')]
    assert len(ids)==len(set(ids))
    topics={e.get('id') for e in parsed.iter(f'{{{C}}}topic')}
    relationships=list(parsed.iter(f'{{{C}}}relationship'))
    for rel in relationships:assert rel.get('end1') in topics and rel.get('end2') in topics
print(json.dumps({'file':str(output),'sheets':len(parsed.findall(f'{{{C}}}sheet')),'topics':len(topics),'relationships':len(relationships),'zip_xml_reference_checks':'passed'},ensure_ascii=False))
