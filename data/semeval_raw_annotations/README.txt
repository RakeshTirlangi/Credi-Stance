SemEval-2016 Task 6 Stance and Sentiment Dataset
Raw Stance Annotations
June 2019
Copyright (C) 2019 National Research Council Canada (NRC)
----------------------------------------------------------------


Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


Terms of Use: 
-------------------------------------------------

1. If you use this dataset, cite the paper below:

Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry, C. SemEval-2016 Task 6: Detecting Stance in Tweets. Proceedings of the International Workshop on Semantic Evaluation (SemEval-2016), San Diego, California, 2016.

2. Do not redistribute the data. Direct interested parties to this page: http://www.saifmohammad.com/WebPages/StanceDataset.htm

3. National Research Council Canada (NRC) disclaims any responsibility for the use of the dataset and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.


The file contains raw stance annotations for tweets used in SemEval-2016 Task 6 'Detecting Stance in Tweets'. Details about this dataset are available in the following paper:

Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry, C. SemEval-2016 Task 6: Detecting Stance in Tweets. Proceedings of the International Workshop on Semantic Evaluation (SemEval-2016), San Diego, California, 2016.


************************************************
File Format
************************************************

Tha annotation file has the following format:
<Worker ID>,<Instance ID>,<Target>,<Tweet>,<Stance>,<Opinion towards>

where

<Worker ID> is an ID for a crowd worker; to preserve privacy, we replaced the CrowdFlower worker IDs with sequential IDs, but kept the same ID for all annotations performed by a particular worker;

<Instance ID> is an ID for an annotated tweet; the IDs are the same that were used in the SemEval-2016 Task 6 training and test datasets; instances without an ID were not used in the shared task (inter-annotator agreement on these instances is less than 60%);

<Target> is the entity of interest; there are 5 possible targets: "Atheism", "Donald Trump", "Feminist Movement", "Hillary Clinton", and "Legalization of Abortion";

<Tweet> is a tweet text;

<Stance> is a worker's answer to the following question: 'From reading the tweet, which of the options below is most likely to be true about the tweeter’s stance or outlook towards the target' 

The possible answers are:
FAVOR: We can infer from the tweet that the tweeter supports the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else)
AGAINST: We can infer from the tweet that the tweeter is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else)
NEUTRAL: We can infer from the tweet that the tweeter has a neutral stance towards the target (e.g., the tweeter supports the target entity to some extent, but is also against it to some extent)
NONE: none of the above.

<Opinion towards> is a worker's answer to the following question: 'From reading the tweet, which of the options below is most likely to be true about the focus of opinion/sentiment in the tweet' 

The possible answers are:
TARGET: The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target
OTHER: The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target
NO ONE: The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)


(Note: for the target 'Climate Change is a Real Concern', tweets were annotated internally and are not included in this file.)


************************************************
More Information
************************************************

Mohammad, S., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry, C. SemEval-2016 Task 6: Detecting Stance in Tweets. Proceedings of the International Workshop on Semantic Evaluation (SemEval-2016), San Diego, California, 2016.

Mohammad, S., Sobhani, P., Kiritchenko, S. (2017). Stance and Sentiment in Tweets. ACM Transactions on Internet Technology, 17(3), 2017.



************************************************
CONTACT INFORMATION
************************************************
Saif M. Mohammad
Senior Research Officer, National Research Council Canada
email: saif.mohammad@nrc-cnrc.gc.ca
phone: +1-613-993-0620
