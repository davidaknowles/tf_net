## Download competition data for the
## ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge
import synapseclient
from synapseclient import Project, Folder, File
import sys
import os

syn = synapseclient.Synapse()

#syn.login(email = 'myemail', password = 'mypassword')

folder_ids = { 'syn6176232' : 'DNase/' ,
               'syn6176231' : 'gene_expression/' ,
               'syn6181335' : 'labels/' }
               
DATADIR=os.environ["DREAM_ENCODE_DATADIR"]

other_files=[ 'syn6184309', # genome sequence
              'syn6401000', # ladder regions
              'syn6184308', # test regions
              'syn6184317' ] # train regions

if not os.path.isdir(DATADIR): os.mkdir(DATADIR)

for f in other_files:
    syn.get(f, downloadLocation=DATADIR)
               
for folder_id in folder_ids:

    # Get folder
    folder = syn.get(folder_id)
    print 'Downloading contents of %s folder (%s)\n' % (folder.name, folder.id,)

    # Query for child entities
    query_results = syn.query('select id,name from file where parentId=="%s"' % folder_id)

    downloadLocation=DATADIR + folder_ids[folder_id]
    if not os.path.isdir(downloadLocation): os.mkdir(downloadLocation)
    
    # Download all data files
    for entity in query_results['results']:
        print '\tDownloading file: ', entity['file.name']
        syn.get(entity['file.id'], downloadLocation=downloadLocation)
        
print 'Download complete!'

