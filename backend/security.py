"""Remove credential-shaped values from catalog metadata."""
import re
SECRET_KEY=re.compile(r'(?:password|passwd|secret|access[_-]?token|api[_-]?key|authorization|private[_-]?key)',re.I)
ASSIGN=re.compile(r'((?:password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]\s*)(?:"[^"]*"|\x27[^\x27]*\x27|[^\s,;]+)',re.I)
FLAG=re.compile(r'((?:--password|--passwd|--token|--api-key|sshpass\s+-p)\s+)(?:"[^"]*"|\x27[^\x27]*\x27|\S+)',re.I)
def scrub(value):
    if isinstance(value,dict):return {k:'[redacted]' if SECRET_KEY.search(str(k)) else scrub(v) for k,v in value.items()}
    if isinstance(value,list):return [scrub(v) for v in value]
    if isinstance(value,str):return FLAG.sub(r'\1[redacted]',ASSIGN.sub(r'\1[redacted]',value))
    return value
