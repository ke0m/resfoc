import random, string
import time
import subprocess

#the job state:
#        E -  Job is exiting after having run. (can be stuck here if not write permissions)
#        H -  Job is held.
#        Q -  job is queued, eligable to run or routed.
#        R -  job is running.
#        T -  job is being moved to new location.
#        W -  job is waiting for its execution time
#       (-a option) to be reached.
#        S -  (Unicos only) job is suspend.

class pbsjob(object):
  """ A class for keeping track of jobs """

  def __init__(self,logpath=".",user='joseph29',verb=False):
    """  Constructor of job class """
    self.jobid = self.id_generator() # Unique job id
    self.subids = {}                 # Submission Ids (job could be across two queues)
    self.subids['sep'] = None; self.subids['default'] = None
    self.status = {}                 # Job status for each queue
    self.status['sep'] = None; self.status['default'] = None
    self.times= {}                   # Job time for each queue
    self.times['sep'] = None; self.times['default'] = None
    self.nodes = {}                  # Nodes names for each job
    self.nodes['sep'] = None; self.nodes['default'] = None
    self.user = user                 # User id
    self.logpath = logpath           # Path for writing log files
    self.verb = verb                 # verbosity flag
    self.nsub = 0                    # Number of attempted submissions
    self.outfile = None              # Stdout log file
    self.errfile = None              # Stderr log file

  def submit(self,name,cmd,nprocs=16,queue='sep',sleep=1):
    """ Submits the job with command to the cluster """
    # First check if the job is running
    if('R' in self.status.values()):
      if(self.verb): print("Job %s is currently running. Not submitting..."%(self.jobid))
      # Also check if it is another queue
      if('Q' in self.status.values()):
        # Gets the key from the value
        q4qdel = list(self.status.keys())[list(self.status.values()).index('Q')]
        q4qkep = list(self.status.keys())[list(self.status.values()).index('R')]
        if(self.verb): print("Killing %s queued in %s and leaving running in %s"%(self.jobid,q4qdel,q4qkep))
        sp = subprocess.check_call("qdel %s"%(self.subids[q4qdel]),shell=True)
      return
    # Check if job is waiting
    if(self.status['sep'] == 'Q' and self.status['default'] == 'Q'):
      if(self.verb): print("Job is waiting in both queues. Not submitting...")
      return
    # Submit to other queue if already waiting
    if(self.status[queue] == 'Q'):
      if(self.verb): print('Job %s waiting in queue %s.'%(self.jobid,queue))
      queue = [key for key in self.status.keys() if key != queue][0]
      if(self.verb): print('Submitting job %s to queue %s'%(self.jobid,queue))

    # Build output log files
    self.outfile = self.logpath + '/' + name + self.jobid + '_out.log'
    self.errfile = self.logpath + '/' + name + self.jobid + '_err.log'
    # Create the PBS script
    pbsout = """#! /bin/tcsh

#PBS -N %s
#PBS -l nodes=1:ppn=%d
#PBS -q %s
#PBS -o %s
#PBS -e %s
cd $PBS_O_WORKDIR
#
%s
#
# End of script"""%(name+self.jobid,nprocs,queue,self.outfile,self.errfile,cmd)
    # Write the script to file
    script = name + self.jobid + ".sh"
    with open(script,'w') as f:
      f.write(pbsout)

    # Submit the script
    sub = "qsub %s > %s"%(script,self.jobid)
    #if(self.verb): print(sub)
    time.sleep(sleep)
    sp = subprocess.check_call(sub,shell=True)
    # Get the submission id and remove the file
    with open(self.jobid,'r') as f:
      self.subids[queue] = f.readlines()[0].rstrip()
    sp = subprocess.check_call('rm %s'%(self.jobid),shell=True)

    # Keep track of submissions
    self.nsub += 1

    return

  def getstatus(self):
    """ Gets the status of the job """
    # Call qstat
    ujobs = "qstat -u %s -n -1 | grep %s > qstat.out"%(self.user,self.jobid)
    #if(self.verb): print(ujobs)
    sp = subprocess.check_call(ujobs,shell=True)
    # Parse the output by the job id and the user
    with open('qstat.out', 'r') as f:
      for line in f.readlines():
        if(self.user in line and self.jobid in line):
          attrs = line.split()
          # Calling exiting completed for now
          if(attrs[9] == 'E'):
            attrs[9] = 'C'
          # Calling held queued for now
          elif(attrs[9] == 'H'):
            attrs[9] = 'Q'
          # Set the status for the queue
          self.status[attrs[2]] = attrs[9]
          # Get the time elapsed and the node id
          if(attrs[9] == 'R'):
            self.times[attrs[2]] = self.toseconds(attrs[10])
            self.nodes[attrs[2]] = attrs[-1][:6]

    return self.status

  def getstatus_fast(self,qstatin):
    """ A faster way of getting job status """
    for line in qstatin:
      if(self.user in line and self.jobid in line):
        attrs = line.split()
        # Calling exiting completed for now
        if(attrs[9] == 'E'):
          attrs[9] = 'C'
        # Calling held queued for now
        elif(attrs[9] == 'H'):
          attrs[9] = 'Q'
        # Set the status for the queue
        self.status[attrs[2]] = attrs[9]
        # Get the time elapsed
        if(attrs[9] == 'R'):
          self.times[attrs[2]] = self.toseconds(attrs[10])
          self.nodes[attrs[2]] = attrs[-1][:6]

    return self.status

  def success(self,flag):
    """ Checks if a completed job was successful """
    if('C' not in self.status.values()):
      if(self.verb): print('Job not yet completed. Exiting function...')
      return
    else:
      # Read in stdout
      with open(self.outfile,'r') as f:
        for line in f.readlines():
          if(flag in line):
            return True

      return False

  def delete(self,queue=None):
    """ Deletes the job from the specified queue """
    if(queue != None):
      sp = subprocess.check_call("qdel %s"%(self.subids[queue]),shell=True)
    else:
      # Remove from both queues
      sp = subprocess.check_call("qdel %s"%(self.subids['sep']),shell=True)
      sp = subprocess.check_call("qdel %s"%(self.subids['default']),shell=True)

  #TODO: wrap qdels with a try catch
  def cleanup(self):
    """ Removes one of two running jobs based on the time """
    if(self.times['sep'] > self.times['default']):
      if(self.verb):
        print("Job %s is running in sep (%d s) and default (%d s). Deleting in default..."%
          (self.jobid,self.times['sep'],self.times['default']))
      sp = subprocess.check_call("qdel %s"%(self.subids['default']),shell=True)
    elif(self.times['sep'] < self.times['default']):
      if(self.verb):
        print("Job %s is running in sep (%d s) and default (%d s). Deleting in sep..."%
          (self.jobid,self.times['sep'],self.times['default']))
      sp = subprocess.check_call("qdel %s"%(self.subids['sep']),shell=True)
    else:
      if(self.verb): print("Job %s is running in sep (%d s) and default (%d s). Deleting in default..."%
          (self.jobid,self.times['sep'],self.times['default']))
      sp = subprocess.check_call("qdel %s"%(self.subids['default']),shell=True)

  def toseconds(self,tstring):
    """ Converts the time from qstat to seconds """
    tsplit = tstring.split(':')
    ours = int(tsplit[0]); mins = int(tsplit[1]); secs = int(tsplit[2])
    return ours*3600 + mins*60 + secs

  def id_generator(self,size=6, chars=string.ascii_uppercase + string.digits):
    """ Creates a random string with uppercase letters and integers """
    return ''.join(random.choice(chars) for _ in range(size))

