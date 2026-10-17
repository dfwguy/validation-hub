"""Faithful extraction of measured Confluence tables from retained page evidence.

Only recognized, explicitly measured table schemas become references. Other tables,
plots and narrative expectations remain searchable source evidence.
"""
from html.parser import HTMLParser
import gzip,hashlib,json,re,math
from pathlib import Path
from . import beacon_nccl


class TableParser(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True);self.tables=[];self.table=None;self.row=None;self.cell=None;self.text=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='table':self.table={'rows':[],'context':' '.join(self.text)[-8000:]}
  elif tag=='tr' and self.table is not None:self.row=[]
  elif tag in ('td','th') and self.row is not None:self.cell={'text':'','links':[],'images':[],'header':tag=='th'}
  elif tag=='a' and self.cell is not None and attrs.get('href'):self.cell['links'].append(attrs['href'])
  elif tag=='ri:attachment' and self.cell is not None:self.cell['images'].append(attrs.get('ri:filename'))
  elif tag in ('br','p') and self.cell is not None:self.cell['text']+=' '
 def handle_data(self,data):
  self.text.append(data)
  if self.cell is not None:self.cell['text']+=data
 def handle_endtag(self,tag):
  if tag in ('td','th') and self.cell is not None:
   self.cell['text']=' '.join(self.cell['text'].split());self.row.append(self.cell);self.cell=None
  elif tag=='tr' and self.row is not None:
   self.table['rows'].append(self.row);self.row=None
  elif tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None


def parse_tables(html):
 parser=TableParser();parser.feed(html);return parser.tables


def load_page(page_id,evidence_dir=None):
 from . import config
 evidence_dir=evidence_dir or config.evidence_root()
 root=Path(evidence_dir);page_id=str(page_id)
 with gzip.open(root/(page_id+'.page.json.gz'),'rt') as stream:page=json.load(stream)
 html=page.get('body',{}).get('storage',{}).get('value')
 if html is None:html=(root/(page_id+'.html')).read_text()
 links=page.get('_links',{});url=links.get('base',config.ATLASSIAN_SITE)+links.get('webui','/pages/viewpage.action?pageId='+page_id)
 return page,html,url


def _number(value):
 try:
  number=float(value.replace(',','').strip());return number if math.isfinite(number) else None
 except (ValueError,AttributeError):return None


def _declared(context,pattern):
 values=set(re.findall(pattern,context,re.I))
 return next(iter(values)) if len(values)==1 else None


def _bytes(value):
 if value is None:return None
 match=re.fullmatch(r'([\d,]+)\s*(B|KiB|MiB|GiB|KB|MB|GB)?',str(value).strip(),re.I)
 if not match:return None
 factor={'b':1,'kib':1024,'mib':1024**2,'gib':1024**3,'kb':1000,'mb':1000**2,'gb':1000**3}[str(match[2] or 'B').lower()]
 return int(match[1].replace(',',''))*factor


def extract_page(page,html,source_url):
 """Extract explicit measured columns; retain missing or ambiguous context as unknown.

No customer, model, network recipe, message size or test protocol is built in.
Training candidates are staged separately and are not automatically imported.
 """
 tables=parse_tables(html);records=[];training=[];inventory=[];page_id=str(page['id']);version=page.get('version',{}).get('number')
 sha=hashlib.sha256(html.encode()).hexdigest()
 for ti,table in enumerate(tables):
  rows=table['rows'];headers=[c['text'] for c in rows[0]] if rows else []
  inventory.append({'table_index':ti,'headers':headers,'rows':max(0,len(rows)-1),'image_names':[image for row in rows for cell in row for image in cell['images']]})
  context=table['context']
  gpu=_declared(context,r'\b(?:GPU|GPU model|Hardware)\s*[:=]\s*([A-Za-z0-9_-]+)\b')
  per_node=_declared(context,r'\b(\d+)\s+GPUs/node\b')
  nccl_version=_declared(context,r'\bNCCL(?: version)?\s*[:=]?\s*(\d+\.\d+\.\d+)\b')
  fabric=_declared(context,r'\bFabric\s*[:=]\s*([A-Za-z0-9_-]+)\b')
  protocol={}
  for field in ('min_bytes','max_bytes','iterations','warmup','registration','stepfactor'):
   value=_declared(context,r'\b'+field+r'\s*[:=]\s*([\d,]+\s*(?:GiB|MiB|KiB|GB|MB|KB|B)?)(?=[;,\s]|$)')
   if value is not None:protocol[field]=_bytes(value) if field.endswith('_bytes') else _number(value)
  training_schema=all(key in headers for key in ('Scale','Result TFLOPS/GPU'))
  nccl_schema=all(key in headers for key in ('Nodes','Collective','algBW','busBW'))
  for ri,cells in enumerate(rows[1:],1):
   if len(cells)!=len(headers):continue
   row={key:cell['text'] for key,cell in zip(headers,cells)}
   provenance={'page_id':page_id,'version':version,'html_sha256':sha,'table_index':ti,'row_index':ri,'source_row':row,
      'links':[link for cell in cells for link in cell['links']]}
   if training_schema:
    scale=re.fullmatch(r'(\d+) GPUs',row['Scale']);measured=_number(row['Result TFLOPS/GPU'])
    if scale and measured is not None:
     training.append({'id':f'{page_id}:table{ti}:row{ri}','source_url':source_url,'gpu':row.get('GPU') or gpu,'gpus':int(scale[1]),
       'workload':row.get('Workload') or _declared(context,r'\bWorkload\s*[:=]\s*([^;\n]+)'),
       'model':row.get('Model') or _declared(context,r'\bModel\s*[:=]\s*([^;\n]+)'),
       'model_size':row.get('Model size') or _declared(context,r'\bModel size\s*[:=]\s*([^;\n]+)'),
       'tflops_gpu':measured,'precision':row.get('Precision') or _declared(context,r'\bPrecision\s*[:=]\s*([A-Za-z0-9_-]+)'),
       'release':row.get('Release') or None,'placement_scope':row.get('Placement'),'source':'NVIDIA Confluence',
       'reference_kind':'measured','is_measured':True,'page_id':page_id,'page_version':version,
       'source_provenance':dict(provenance,measured_column='Result TFLOPS/GPU'),
       'caveats':['Only the Result TFLOPS/GPU column is measured evidence; comparison baseline columns remain annotations.','Missing configuration fields remain unknown.']})
   if not nccl_schema:continue
   # Unit and actual payload must be explicit; never assume a sweep endpoint.
   units=row.get('Bandwidth unit') or ('GB/s' if 'GB/s' in context else None)
   if units!='GB/s':continue
   nodes=_number(row['Nodes'].removesuffix('N'));bus=_number(row['busBW']);alg=_number(row['algBW'])
   if nodes is None or int(nodes)!=nodes or nodes<=0 or bus is None:continue
   size_match=re.search(r'max size ([\d,]+) B',row.get('Assessment',''))
   size=_bytes(row.get('Message bytes') or row.get('message_bytes'))
   if size_match:size=int(size_match[1].replace(',',''))
   final_row='final/max-size row' in context
   if size is None and final_row:size=protocol.get('max_bytes')
   if size is None or size<=0:continue
   placement=row.get('Memory placement') or ('out_of_place' if 'out-of-place final/max-size row' in context else None)
   provenance.update(measured_columns=['algBW','busBW'],caveats=['Execution completion alone does not establish correctness.','Comparison baseline and theoretical columns are annotations only.','Opaque source profiles do not establish parallelism or split masks.'])
   records.append({'source_record_id':f'{page_id}:table{ti}:row{ri}','page_id':page_id,'page_version':version,
     'page_title':page.get('title'),'table_index':ti,'row_index':ri,'gpu':row.get('GPU') or gpu,
     'gpus':int(nodes)*int(per_node) if per_node else None,'nodes':int(nodes),'fabric':row.get('Fabric') or fabric,
     'nic_model':row.get('NIC model'),'nccl_version':row.get('NCCL version') or nccl_version,'collective':row['Collective'],
     'placement_scope':row.get('Placement'),'source_test_profile':row.get('TP'),'optimization_profile':row.get('OP'),
     'execution_status':row.get('Execution'),'reference_kind':'measured','is_measured':True,
     'confidence':'published measured table; unknown context retained','test_protocol':protocol,
     'metrics':[{'message_bytes':size,'placement':placement,'busbw_gbps':bus,'algbw_gbps':alg,'bandwidth_unit':units}],
     'message_size_source':'explicit row size' if size_match or row.get('Message bytes') or row.get('message_bytes') else 'explicit documented final/max-size endpoint',
     'source_provenance':provenance,'source_recipe_scope':{'source_page':source_url},
     'unit_provenance':{'bandwidth':units,'source':'explicit table unit or preceding source context'},'raw_asset_sha256':sha})
 return {'page_id':page_id,'page_version':version,'source_url':source_url,'html_sha256':sha,'tables':inventory,'references':records,'training_references':training,
   'limitations':['Unrecognized tables, image plots and narrative expectations are evidence only; no numeric targets inferred.']}


def index_page(page_id,evidence_dir=None,db_path=None,import_measured=False):
 page,html,url=load_page(page_id,evidence_dir);result=extract_page(page,html,url)
 if import_measured and result['references']:
  result['receipt']=beacon_nccl.import_records(result['references'],url,db_path,metadata={'source_dataset':'confluence measured NCCL tables','page_id':str(page_id),'page_version':result['page_version'],'reported_total':len(result['references']),'complete_all_pages':True})
 return result
