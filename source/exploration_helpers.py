import numpy as np
import pandas as pd


def get_membership_length(df):
    """Calculates the membership length back from the latest found became_member_on date and
    deletes the became_member_on column.
    """
    # converting the 'became_member_on' column into date:
    df.became_member_on = pd.to_datetime(df.became_member_on.astype(str), format='%Y%m%d')

    # latest date in the column:
    max_date = df.became_member_on.max()

    # inserting an extra column called 'membership_length' and filling it up with the difference.
    # (date_1 - date_2) returns a Timedelta object. To get the day difference in days Timedelta.days is needed.
    df['membership_length'] = (max_date - df.became_member_on).apply(lambda x: x.days)

    # removing became_member_on field
    df = df.drop('became_member_on', axis='columns')

    return df


def extract_transcript_values(transcript):
    """Extracts the dictionary of the transcript DataFrame's value column. 
    Also somer offer id are stored under 'offer id' and some are under 'offer_id'. These columns are mergerd.
    """
    transcript = transcript.join(pd.DataFrame.from_records(transcript.pop('value')))
    transcript.offer_id.update(transcript.pop('offer id'))

    return transcript


def get_offer_stats(transcript, profile, portfolio):
    """This function calculates for each person in the profile dataframe, how many times a person have
    viewed, viewed and completed and not viewed but completed offers during the record period.
    It creates 3 extra columns respectively in the profile DataFrame and stores the results there.
    """
    # adding some extra columns to the profile dataframe:
    if 'num_viewed' not in profile.columns:
        profile.insert(loc=len(profile.columns), column='num_viewed', value=np.zeros_like(profile.index))
    
    if 'num_viewed_completed' not in profile.columns:
        profile.insert(loc=len(profile.columns), column='num_viewed_completed', value=np.zeros_like(profile.index))
    
    if 'num_not_viewed_completed' not in profile.columns:
        profile.insert(loc=len(profile.columns), column='num_not_viewed_completed', value=np.zeros_like(profile.index))

    # pre calculating the boolean masks of the indices which contain 'offer viewed' and 'offer completed' values:
    viewed_mask = (transcript.event == 'offer viewed')
    completed_mask = (transcript.event == 'offer completed')

    # calculating for every person the number of viewed, viewed and completed and not viewed but completed offers:
    for i in profile.index:
        person = profile.id[i]
        person_mask = (transcript.person == person)   
        
        viewed_indices = transcript.index[person_mask & viewed_mask]
        completed_indices = transcript.index[person_mask & completed_mask]
        
        # checking for completed offers whether they were completed before or after they were viewed:
        viewed_ids = transcript.offer_id[viewed_indices].values
        viewed_times = transcript.time[viewed_indices].values
        
        completed_ids = transcript.offer_id[completed_indices].values
        completed_times = transcript.time[completed_indices].values
        
        num_viewed_completed = 0
        num_not_viewed_completed = 0
        for j, offer_id in enumerate(completed_ids):
            time_it_was_completed = completed_times[j]
            
            # length of this actual offer:
            length_of_offer = portfolio.duration[portfolio.id == offer_id].values[0]
            
            if offer_id in viewed_ids:
                # times when the person has viewed an offer with this id:
                times_it_was_viewed = viewed_times[viewed_ids == offer_id]
                
                # times when these offers expired:
                times_end = [time + length_of_offer for time in times_it_was_viewed]
                
                # binary masks for checking whether any of the completions were after the offer
                # was viewed and before it expired:
                completed_after_viewed = (times_it_was_viewed <= time_it_was_completed)
                completed_before_ended = (times_end >= time_it_was_completed)
                
                if (completed_after_viewed & completed_before_ended).any():
                    num_viewed_completed += 1
                else:
                    num_not_viewed_completed += 1
            else:
                num_not_viewed_completed += 1
        
        # filling up the dataframe of the person:
        profile.num_viewed[i] = (person_mask & viewed_mask).sum()
        profile.num_viewed_completed[i] = num_viewed_completed
        profile.num_not_viewed_completed[i] = num_not_viewed_completed

        # printing progress:
        if (i%1000 == 0):
            print('{} out of {} is done'.format(i, len(profile.index)))
    
    return profile
    