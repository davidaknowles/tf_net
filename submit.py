import synapseclient

import os
DATADIR=os.environ["DREAM_ENCODE_DATADIR"]

syn = synapseclient.Synapse()

syn.login(email = '', password = '')

evaluation = syn.getEvaluation(7366344) # Round 2 final will be 7373880
submissions_folder = "need_to_set_this" 

import glob
submission_filenames=glob.glob(DATADIR + "submissions/L*.tab.gz")

for filename in submission_filenames:
        f = synapseclient.File( filename, parent=submissions_folder )
        entity = syn.store(f)
        syn.submit(evaluation, entity, name="first_submission") # , team="Team Awesome")
