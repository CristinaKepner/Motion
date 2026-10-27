This Zenodo repository contains the main dataset for the GENEA Challenge 2023, which is based on the Talking With Hands 16.2M dataset.

Notation:
Please take note of the following nomenclature when reading this document:
* main agent refers to the speaker in the dyadic interaction for which the systems generated motions. 
* interlocutor refers to the speaker in front of the main agent. 

Contents:
The “genea2023_trn" and "genea2023_val" zip files contain audio files (in WAV format), time-aligned transcriptions (in TSV format), and motion files (in BVH format) for the training and validation datasets, respectively.
The "genea2023_test" zip file contains audio files (in WAV format) and transcriptions (in TSV format) for the test set, but no motion. The corresponding test motion is available at:
https://zenodo.org/record/8146028
Each zip file also contains a "metadata.csv" file that contains information for all files regarding the speaker ID and whether or not the motion files contain finger motion.
Note that the speech audio in the data sometimes has been replaced by silence for the purpose of anonymisation.
In the test set, files with indices from 0 to 40 correspond to "matched" interactions (the core test set), where main agent and interlocutor data come from the same conversation, whilst file indices from 41 to 69 correspond to "mismatched" interactions (the extended test set), where main agent and interlocutor data come from different conversations.

Folder structure:
* main-agent/ (main agent): Encapsulates BVH, TSV, WAV data for the main agent.
* interloctr/ (interlocutor): Encapsulates BVH, TSV, WAV data for the interlocutor.
* bvh/ (motion): Time-aligned 3D full-body motion-capture data in BVH format from a speaking and gesticulating actor. Each file is a single person, but each data sample contains files for both the main agent and the interlocutor. 
* wav/ (audio): Recorded audio data in WAV format from a speaking and gesticulating actor with a close-talking microphone. Parts of the audio recordings have been muted to omit personally identifiable information. 
* tsv/ (text): Word-level time-aligned text transcriptions of the above audio recordings in TSV format (tab-separated values). For privacy reasons, the transcriptions do not include references to personally identifiable information, similar to the audio files. 

Data processing scripts:
We provide a number of optional scripts for encoding and processing the challenge data:
Audio: Scripts for extracting basic audio features, such as spectrograms, prosodic features, and mel-frequency cepstral coefficients (MFCCs) can be found at this link.
Text: A script to encode text transcriptions to word vectors using FastText is available here: tsv2wordvectors.py
Motion: If you wish to encode the joint angles from the BVH files to and from an exponential map representation, you can use scripts by Simon Alexanderson based on the PyMo library, which are available here:
* bvh2features.py
* features2bvh.py

Attribution:
If you use this material, please cite our latest paper on the GENEA Challenge 2023. At the time of writing (2023-07-25) this is our ACM ICMI 2023 paper:
* Taras Kucherenko, Rajmund Nagy, Youngwoo Yoon, Jieyeon Woo, Teodor Nikolov, Mihail Tsakov, and Gustav Eje Henter. 2023. The GENEA Challenge 2023: A large-scale evaluation of gesture generation models in monadic and dyadic settings. In Proceedings of the ACM International Conference on Multimodal Interaction (ICMI ’23). ACM.
Also, please cite the paper about the original dataset from Meta Research:
* Gilwoo Lee, Zhiwei Deng, Shugao Ma, Takaaki Shiratori, Siddhartha S. Srinivasa, and Yaser Sheikh. 2019. Talking With Hands 16.2M: A large-scale dataset of synchronized body-finger motion and audio for conversational motion analysis and synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV ’19). IEEE, 763–772.
 
The motion and audio files are based on the Talking With Hands 16.2M dataset at https://github.com/facebookresearch/TalkingWithHands32M/. The material is available under a CC BY NC 4.0 Attribution-NonCommercial 4.0 International license, with the text provided in LICENSE.txt.
 
To find more GENEA Challenge 2023 material on the web, please see:
* https://genea-workshop.github.io/2023/challenge/
 
If you have any questions or comments, please contact:
* The GENEA Challenge organisers <genea-challenge@googlegroups.com>