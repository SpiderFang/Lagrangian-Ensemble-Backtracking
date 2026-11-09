import subprocess,re,sys,json
pdf=sys.argv[1]; out=sys.argv[2]
n=int(re.search(r'Pages:\s+(\d+)',subprocess.run(['pdfinfo',pdf],capture_output=True,text=True).stdout).group(1))
pages=[subprocess.run(['pdftotext','-f',str(i),'-l',str(i),'-layout',pdf,'-'],capture_output=True,text=True).stdout for i in range(1,n+1)]
il=subprocess.run(['pdfimages','-list',pdf],capture_output=True,text=True).stdout.split('\n')[2:]
imgs=[]
for l in il:
    f=l.split()
    if len(f)>3: imgs.append([int(f[0]),f[2],int(f[3]),int(f[4])])
json.dump(dict(n=n,pages=pages,imgs=imgs),open(out,'w'),ensure_ascii=False)
print(n,len(imgs))
