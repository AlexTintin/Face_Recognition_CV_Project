This folder contains Python described classes and objects in order to build efficient data generators. The goals are:

Use Queues to limit RAM usage to what is needed.
Use Multiprocessing to assist Queue loading
Maintain all dataset operations internally
Return partial set at the end of an epoch rather than error
Return specifically defined error after end of epoch to signal end

Below are the DataSet generators included: