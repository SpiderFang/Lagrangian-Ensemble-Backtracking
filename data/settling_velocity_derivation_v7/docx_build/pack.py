import sys; sys.path.insert(0,'/tmp/w')
from pk import rewrite
rewrite('/tmp/w/orig.docx', sys.argv[2], open(sys.argv[1],'rb').read())
