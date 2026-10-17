"""GET-only Confluence client. Credentials never enter saved source evidence."""
import base64,gzip,hashlib,json,os,re,uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode,urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler

from . import config
SITE=config.ATLASSIAN_SITE

class _SameOrigin(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,url):
        if urlsplit(request.full_url).netloc!=urlsplit(url).netloc:raise ValueError('Cross-origin authenticated redirect refused')
        return super().redirect_request(request,fp,code,message,headers,url)

class ConfluenceReader:
    def __init__(self,email=None,token_file=None,cloud_id=None):
        self.email=email or os.environ.get('ATLASSIAN_EMAIL')
        if not self.email:raise ValueError('Set ATLASSIAN_EMAIL or supply the token owner email')
        self.token_file=Path(token_file or os.environ.get('ATLASSIAN_TOKEN_FILE',str(Path.home()/'.config/validation-hub/atlassian.token')))
        tenant=cloud_id or os.environ.get('ATLASSIAN_CLOUD_ID')
        if not tenant or not re.fullmatch(r'[0-9a-fA-F-]{36}',tenant):raise ValueError('Set ATLASSIAN_CLOUD_ID to the authorized tenant UUID')
        self.base='https://api.atlassian.com/ex/confluence/'+tenant+'/wiki'

    def get(self,path,query=None):
        parsed=urlsplit(path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith(('/api/v2/','/rest/api/')):raise ValueError('A relative Confluence API path is required')
        if self.token_file.stat().st_mode & 0o077:raise ValueError('Atlassian token file must be private (mode 600)')
        secret=self.token_file.read_text().strip()
        if not secret:raise ValueError('Atlassian token file is empty')
        auth='Basic '+base64.b64encode((self.email+':'+secret).encode()).decode()
        url=self.base+path+('?' + urlencode(query) if query else '')
        request=Request(url,headers={'Authorization':auth,'Accept':'application/json'},method='GET')
        try:
            with build_opener(_SameOrigin()).open(request,timeout=60) as response:
                data=response.read(50*1024**2+1)
                if len(data)>50*1024**2:raise ValueError('Confluence response exceeds its size bound')
                return json.loads(data)
        except HTTPError as error:
            # Never echo arbitrary response headers or credential-bearing requests.
            raise ValueError('Confluence read failed with HTTP '+str(error.code)) from None

    def search(self,cql,limit=100,max_pages=100):
        path='/rest/api/content/search';query=dict(cql=cql,limit=min(100,max(1,limit)),expand='version,space');seen=set();items={};pages=0
        while True:
            result=self.get(path,query);pages+=1
            for row in result.get('results',[]):items[row['id']]=row
            next_link=result.get('_links',{}).get('next')
            if not next_link:break
            if next_link in seen or pages>=max_pages:raise ValueError('Incomplete or repeated Confluence search pagination')
            seen.add(next_link);parts=urlsplit(next_link)
            if parts.netloc and parts.netloc!=urlsplit(SITE).netloc:raise ValueError('Cross-origin pagination refused')
            path=parts.path.removeprefix('/wiki')+('?'+parts.query if parts.query else '');query=None
        return dict(query=cql,complete=True,pages=pages,count=len(items),items=list(items.values()))

    def save_page(self,page_id,folder):
        if not re.fullmatch(r'\d+',str(page_id)):raise ValueError('Numeric Confluence page ID required')
        result=self.get('/api/v2/pages/'+str(page_id),{'body-format':'storage'})
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(result,separators=(',',':')).encode();pid=str(page_id)
        outputs={pid+'.page.json.gz':gzip.compress(payload,mtime=0),pid+'.html':result.get('body',{}).get('storage',{}).get('value','').encode(),
                 pid+'.metadata.json':json.dumps({k:v for k,v in result.items() if k!='body'},indent=2).encode()}
        for name,data in outputs.items():
            temporary=folder/(uuid.uuid4().hex+'.partial');temporary.write_bytes(data);temporary.replace(folder/name)
        return dict(id=pid,title=result['title'],version=result.get('version',{}).get('number'),sha256=hashlib.sha256(payload).hexdigest())
