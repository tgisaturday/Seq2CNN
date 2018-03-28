import time
import psycopg2 as psql
import sys
import os
import re

def descriptionParser(record):
    
    abstract = record[4]
    if abstract == None:
        return
    abstract = re.sub('<[^>]*>',' ',abstract)
    description = record[5]
    if description == None:
        return
    description = re.sub('<[^>]*>',' ',description)
    
    if '내용없음' in abstract:
        return
    if '내용 없음' in abstract:
        return
    if '요부공개' in description:
        return
    if '요구공개' in description:
        return
    if '요부 공개' in description:
        return
    if '요구 공개' in description:
        return
    if '내용없음' in description:
        return
    if '내용 없음' in description:
        return
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
    if not os.path.isdir('dataset/abstract'):
        os.mkdir('dataset/abstract')
    if not os.path.isdir('dataset/description'):
        os.mkdir('dataset/description')
    
    if not os.path.isdir('dataset/abstract/'+ str(record[1])):
        os.mkdir('dataset/abstract/'+ str(record[1]))
    if not os.path.isdir('dataset/abstract/'+ str(record[1])+'/'+str(record[2])):
        os.mkdir('dataset/abstract/'+ str(record[1])+'/'+str(record[2]))
    if not os.path.isdir('dataset/abstract/'+ str(record[1])+'/'+str(record[2])+'/'+str(record[3])):
        os.mkdir('dataset/abstract/'+ str(record[1])+'/'+str(record[2])+'/'+str(record[3]))
    if not os.path.isdir('dataset/description/'+ str(record[1])):
        os.mkdir('dataset/description/'+ str(record[1]))
    if not os.path.isdir('dataset/description/'+ str(record[1])+'/'+str(record[2])):
        os.mkdir('dataset/description/'+ str(record[1])+'/'+str(record[2]))
    if not os.path.isdir('dataset/description/'+ str(record[1])+'/'+str(record[2])+'/'+str(record[3])):
        os.mkdir('dataset/description/'+ str(record[1])+'/'+str(record[2])+'/'+str(record[3]))
        
    filename = str(record[1])+'/'+str(record[2])+'/'+str(record[3])+'/'+str(record[0])
    fp_abs = open('dataset/abstract/'+ filename + '.txt','w')
    fp_abs.write(abstract)
    fp_abs.close()

    fp_des = open('dataset/description/'+ filename + '.txt','w')
    fp_des.write(description)
    fp_des.close()

if __name__ == '__main__':
    
    query_count = """SELECT count(*) FROM public.application_kr_ipc where unit_doc_id > 'KR1020170000000A-1020170000000';"""
    cur = conn.cursor()
    cur.execute(query_count)
    total = cur.fetchall()
    limit = 1000
    offset = 0
    start = time.time()

    while True:
        sys.stdout.write("\rTotal: %d / %d" % (offset, total[0][0])),
        sys.stdout.flush()
        cur.execute(query,(limit,offset))
        records = cur.fetchall()
        offset += limit
        if records:
            for record in records:
                descriptionParser(record)
        else:
            break

    print("\nExecution time = {0:.5f}".format(time.time() - start))
    cur.close()
    conn.close()




