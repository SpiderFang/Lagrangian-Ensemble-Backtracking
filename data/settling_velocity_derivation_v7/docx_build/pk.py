import zipfile, shutil
def rewrite(src,dst,newxml_bytes,part='word/document.xml'):
    zin=zipfile.ZipFile(src)
    zout=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED)
    for it in zin.infolist():
        data=zin.read(it.filename)
        if it.filename==part: data=newxml_bytes
        ni=zipfile.ZipInfo(it.filename,it.date_time)
        ni.compress_type=it.compress_type; ni.external_attr=it.external_attr; ni.create_system=it.create_system
        zout.writestr(ni,data)
    zout.close()
