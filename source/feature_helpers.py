import numpy as np
import pandas as pd
import os


def get_total_spent(profile_df, transcript_df, cache_file=None):
    """This function calculates for every person in profile_df the total consumption based on transcript_df
    If cache is available, it tries to load from the cache file.
    """

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(cache_file, "rb") as f:
                cache_data = pd.read_csv(f, index_col=0)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            print("Unable to read cache file")
    else:
        print("No cache file was given")

    if cache_data is None:
        # creating a transaction dataframe:
        transactions_df = transcript_df[transcript_df.event == 'transaction']

        if 'total_spent' not in profile_df.columns:
            profile_df.insert(loc=len(profile_df.columns), column='total_spent', value=np.zeros_like(profile_df.index))

        counter = 0
        for person in profile_df.id:
            person_mask = (transactions_df.person == person)
            transactions_of_person = transactions_df.amount[person_mask]
            
            profile_df.total_spent.loc[profile_df.id == person] = transactions_of_person.sum()

            counter += 1
            # printing stats occasionally:
            if (counter%1000 == 0):
                print('{} out of {} done'.format(counter, len(profile_df.index)))

        # saving cache:
        cache_dir = 'cache'
        cache_file = 'profile_total_spent.csv'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        
        with open(os.path.join(cache_dir, cache_file), 'w'):
            profile_df.to_csv(os.path.join(cache_dir, cache_file))
            print('cache was saved to {}'.format(os.path.join(cache_dir, cache_file)))    

    else: 
        # if cache reading was succesful the data of it is returned
        profile_df = cache_data
    
    return profile_df


def remove_outliers(profile, transcript, lower_threshold=0, upper_threshold=300):
    """This funciton identifies the outpliers in the dataset and removes every datapoint which is in connection
    with them.
    A person is considered to be outplier if his/her total_spent value is bigger than a certain upper_threshold or
    smaller than a lower_threshold.
    """
    # identifying the outliers:
    mask_lower = (profile.total_spent <= lower_threshold)
    mask_upper = (profile.total_spent > upper_threshold)
     
    outliers_df = profile.id[mask_upper | mask_lower]    
    
    for outlier in outliers_df:
        # removing the outliers from the transcript data:
        indices_to_drop = transcript.index[transcript.person == outlier]
        transcript = transcript.drop(index=indices_to_drop)
        
        # removing the outliers from the profile data:
        profile = profile.drop(index=profile.index[profile.id == outlier])
        
    
    return profile, transcript


def encode_gender(profile):
    """This function encodes the gender into one-hot-encoding representation.
    It inserts the columns 'F', 'M', 'O', 'U' into the profile dataframe and removes the original
    gender column.
    """
    for column in ['U', 'O', 'M', 'F']:
        if column not in profile.columns:
            profile.insert(loc=1, column=column, value=np.zeros_like(profile.index))

    for gender in ['F', 'M', 'O', 'U']:
        if gender in ['F', 'M', 'O']:
            gender_mask = (profile.gender == gender)
        else:
            gender_mask = profile.gender.isna()
        
        # filling up the appropiate places with ones:
        profile.loc[gender_mask, gender] = 1
        
    # removing the gender column:
    profile = profile.drop(columns='gender')
    
    return profile