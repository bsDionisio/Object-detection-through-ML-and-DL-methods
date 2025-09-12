from torch.utils.tensorboard import SummaryWriter

class Logger:
    #Constructor; It's runs automatically whenever you create a new Logger object
    def __init__(self, path):
        #It creates a writer  object that logs data (scalars, images, histograms, etc.) into the given directory (path)
        self.sw           = SummaryWriter(path)
        self.tag_counters = {}

    #It's used to log a scalar value (a single number, like loss or accuracy) to TensorBoard
    #Input: tag = the name of the metric; value = the actual number you want to log 
    def add_scalar(self, tag, value):
        #Thie dictionary self.tag_counters keeps track of how many times each tag has been logged; 
        # if first time we're logging this tag, initialize its counter to 0
        if tag not in self.tag_counters:
            self.tag_counters[tag] = 0

        #Retrieve the current counter for this tag
        counter = self.tag_counters[tag] 

        #Logs the scalar value to TensorBoard under the given tag
        self.sw.add_scalar(tag, value, global_step=counter)
        #After logging, increment the counter for this tag so the next call uses the next step
        self.tag_counters[tag] += 1

    #This method extends the functionality of add_scalar
    def add_scalars(self, tag_to_value, prefix=''):
        #If prefix is given, append a / to it so the tags are grouped hierarchically in TensorBoard
        if prefix != '':
            prefix = f'{prefix}/'

        #Iterate through each (tag, value) pair in the dictionary
        for tag, value in tag_to_value.items():
            #Prepend the prefix (if any)
            tag = f'{prefix}{tag}'
            #Call self.add_scalar() for each, so that logging is handled with counters automatically
            self.add_scalar(tag, value)