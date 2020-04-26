import numpy as np
import pandas as pd
import os


def get_training_data(profile, transcript, portfolio, target_dir='data', target_file='training_data.csv',
                      cache_dir='cache', pruning_cache='pruned_received_offers.csv'):
    """This function creates the training data from the 3 datasets profile, transcript and portfolio.
    It returns a DataFrame which includes the engineered feature vectors and correspoding labels.
    It also saves the dataframe to a file provided by target_dir and target_file.
    
    The features and their location:
        - gender
        - age
        - income
        - membership length
        - average spent money a day until receiving the offer
        - number of received offers until receiving the offer
        - ratio of viewed / received offers until receiving the offer
        - ratio of completed / viewed offers until receiving the offer
        - ratio of completed / not viewed offers until receiving the offer
        - number of received offers of this type until receiving the offer
        - ratio of viewed / received offers of this type until receiving the offer
        - ratio of completed / viewed offers of this type until receiving the offer
        - ratio of completed / not viewed offers of this type until receiving the offer
        - one-hot-encoding of the currently received offer type
        
    The labels have one of the following values:
        0. not viewed, not completed
        1. not viewed but completed
        2. viewed but not completed
        3. viewed and completed
    """
    
    # received offers:
    received_offers = transcript[transcript.event == 'offer received']
    print('pruning received offers.')
    
    # pruning the offers which came too late:
    if pruning_cache is not None:
        received_offers = pruning_late_offers(received_offers, portfolio,
                                              cache_file=os.path.join(cache_dir, pruning_cache))
    else:
        received_offers = pruning_late_offers(received_offers, portfolio)
    print('pruning is complete.')
    
    # number of remaining received offers and hence training points:
    num_training_points = len(received_offers.index)
    print('number of training points: {}'.format(num_training_points))
    # resetting the index of received_offers
    received_offers.index = list(range(num_training_points))
    
    # column names of the training data dataframe:
    # personal info columns:
    personal_columns = ['F', 'M', 'O', 'U', 'age', 'income', 'membership_length']
    # general offer statistics:
    av_consumption_column = ['av_money_spent']
    # offer statistics:
    offer_stat_columns = [# offer statistics in general:
                          'num_received',
                          'viewed/received', 'completed/viewed', 'completed_not_viewed',
                          # offer statistics about this very type:
                          'num_received_this',
                          'viewed/received_this', 'completed/viewed_this', 'completed_not_viewed_this']
    # offer type cathegory columns:
    offer_columns = ['offer_0', 'offer_1', 'offer_2', 'offer_3', 'offer_4',
                     'offer_5', 'offer_6', 'offer_7', 'offer_8', 'offer_9']
    # label columns:
    label_column = ['label']
    
    # columns of the dataframe:
    columns = personal_columns + av_consumption_column + offer_stat_columns + offer_columns + label_column
    
    # creating the dataframe:
    training_data = pd.DataFrame(columns=columns, index=list(range(num_training_points)))
    
    # creating a dictionary to encode the offer type with one-hot-encoding
    offer_ids = portfolio.id.values
    offer_encode_dict = dict(zip(offer_ids, offer_columns))
    
    # filling up the offer type encoding columns with zeros in advance:
    training_data.loc[:, offer_columns] = np.zeros((num_training_points, len(offer_columns))).astype(np.int8)
    
    # some masks for furhther use later for searching in the transcript data:
    viewed_mask = (transcript.event == 'offer viewed')
    completed_mask = (transcript.event == 'offer completed')
    
    for i, offer in received_offers.iterrows():
        ####################### FEATURE CREATION #######################
        # extracting the info about this received offer:
        person = offer.person
        offer_id = offer.offer_id
        time = offer.time
        
        # filling up the personal data of the datapoint
        personal_data = profile[profile.id == person][personal_columns]
        training_data.loc[i, personal_columns] = personal_data.values
        
        # encoding the offer type into one-hot-encoding:
        offer_type = offer_encode_dict[offer_id]
        training_data.loc[i, offer_type] = 1
        
        # getting all transcript data of this person until this moment:
        actual_transcript = get_data_until_now(offer, transcript)
        
        # average money spent until this day:
        training_data.loc[i, 'av_money_spent'] = get_average_consumption(offer, actual_transcript)
        
        # offer statistics until this day:
        training_data.loc[i, offer_stat_columns] = get_offer_stats(offer, actual_transcript, portfolio,
                                                                   columns=offer_stat_columns).values
        
        ####################### LABEL CREATION #######################
        # here we check in which cathegory the received offer falls.
        
        duration = portfolio.duration[portfolio.id == offer_id].values[0]
        expiration = time + duration
        
        # print('time: {}, duration: {}, expiration: {}'.format(time, duration, expiration))
        
        # mask for transcript elements which are within the expiration date:
        time_mask = (transcript.time > time) & (transcript.time <= expiration)
        
        # viewed and completed offers within this period:
        viewed_offers = transcript[viewed_mask & time_mask]
        viewed_offers_this = viewed_offers[viewed_offers.offer_id == offer_id]
        completed_offers = transcript[completed_mask & time_mask]
        completed_offers_this = completed_offers[completed_offers.offer_id == offer_id]
        
        training_data.loc[i, label_column] = cathegorize_offer(viewed_offers_this, completed_offers_this)
        
        
        
        if (i%1000 == 0) and (i != 0):
            print('{} out of {} training points are complete.'.format(i, num_training_points))
            
    # saving the results in a csv file:
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    with open(os.path.join(target_dir, target_file), 'w'):
        training_data.to_csv(os.path.join(target_dir, target_file))
        print('training dataset was saved to {}'.format(os.path.join(target_dir, target_file)))  
    
    return training_data


def pruning_late_offers(received_offers, portfolio, cache_file=None):
    """This function prunes the offers from the transcript which came too late by checking the expiration date.
    If the expiration date is beyond the 30 day period the received order is erased.
    if the cache_file is not None, it tries to read it first.
    """
    # the deadline in hours:
    deadline = 30 * 24
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(cache_file, "rb") as f:
                cache_data = pd.read_csv(f, index_col=0)
            print("Read pruned data from cache file: ", cache_file)
        except:
            print("Unable to read cache file")
    else:
        print("No cache file was given")
    
    if cache_data is None:
        for i, row in received_offers.iterrows():
            offer_id = row.offer_id
            time = row.time

            offer_duration = portfolio.duration[portfolio.id == offer_id].values
            expiration_time = time + offer_duration

            if expiration_time > deadline:
                received_offers = received_offers.drop(index=i)
                
        # saving cache:
        cache_dir = 'cache'
        cache_file = 'pruned_received_offers.csv'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        with open(os.path.join(cache_dir, cache_file), 'w'):
            received_offers.to_csv(os.path.join(cache_dir, cache_file))
            print('cache was saved to {}'.format(os.path.join(cache_dir, cache_file)))  
    else:
        received_offers = cache_data
            
    
    return received_offers


def get_data_until_now(offer, transcript):
    """This function gives back all the transactional and offer data of a person until the moment 
    the offer held in row is received.
    Returns a DataFrame containing the transactional data in question.
    """
    # extracting the info about this received offer:
    person = offer.person
    time = offer.time
    
    # mask of the data which happened before this event:
    time_mask = (transcript.time < time)
    # mask of the data which is in connection with this person:
    person_mask = (transcript.person == person)
    # overall mask:
    mask = time_mask & person_mask
    
    # returning the relevant data:
    return transcript[mask]


def get_average_consumption(offer, actual_transcript):
    """Returns the average consumption until the offer was received."""
    
    time = float(offer.time)
    transaction_mask = (actual_transcript.event == 'transaction')
    
    total_spent = actual_transcript.amount[transaction_mask].sum()
    
    if (time == 0):
        average_spent = 0
    else:
        average_spent = total_spent / time * 24.0
    
    return average_spent


def get_offer_stats(offer, actual_transcript, portfolio, columns):
    """This function provides the following statistics until the time the offer was received:
    - number of received offers until receiving the offer
    - ratio of viewed / received offers until receiving the offer
    - ratio of completed / viewed offers until receiving the offer
    - ratio of completed not viewed / received offers until receiving the offer
    - number of received offers of this type until receiving the offer
    - ratio of viewed / received offers of this type until receiving the offer
    - ratio of completed / viewed offers of this type until receiving the offer
    - ratio of completed not viewed / received offers of this type until receiving the offer
    
    The columns provided should also have the order as above.
    
    It returns a DataFrame with this info.
    """
    
    # creating the empty dataframe with the column names:
    offer_stats = pd.DataFrame(columns=columns, index=[0])
    
    ################################################################################
    #            STATS ABOUT ALL THE OFFERS IN GENERAL
    ################################################################################
    """This part calculates the stats (number of received, viewed, ...) about the dataset.
    Here every type of offers are considered.
    """
    
    # the number of received offers:
    received_mask = (actual_transcript.event == 'offer received')
    received_offers = actual_transcript[received_mask]
    num_received = received_mask.sum()
    
    # number of viewed offers:
    viewed_mask = (actual_transcript.event == 'offer viewed')
    num_viewed = viewed_mask.sum()
    
    # number of completed offers:
    completed_mask = (actual_transcript.event == 'offer completed')
    completed_offers = actual_transcript[completed_mask]
    num_completed = completed_mask.sum()
    
    # informational type offers can not be completed, so they are not included further in the viewed offers:
    info_ids = ['3f207df678b143eea3cee63160fa8bed', '5a8bc65990b245e5a138643cd4eb9837']
    not_info_mask = (actual_transcript.offer_id != info_ids[0]) & (actual_transcript.offer_id != info_ids[1])
    num_viewed_not_info = (viewed_mask & not_info_mask).sum()
    viewed_not_info_offers = actual_transcript[viewed_mask & not_info_mask]
    
    # number of viewed and completed offers:
    num_viewed_completed = 0
    for i, viewed_offer in viewed_not_info_offers.iterrows():
        offer_id = viewed_offer.offer_id
        time = viewed_offer.time
        duration = portfolio.duration[portfolio.id == offer_id].values[0]
        
        # beginning time of the viewed offer:
        # getting the last received offer of this type before it was viewed:
        mask_before = (received_offers.time <= time)
        mask_id = (received_offers.offer_id == offer_id)
        begin_time = received_offers.time[mask_before & mask_id].max()
        
        # expiration time of the viewed offer:
        expiration_time = begin_time + duration
        
        # checking if there is an offer after viewing and before expiration which was completed
        # and has the same offer_id
        id_mask = (completed_offers.offer_id == offer_id)
        time_mask = (completed_offers.time > time) & (completed_offers.time <= expiration_time)
        
        # viewed_and_completed is True if the offer was completed after viewed and False otherwise:
        viewed_and_completed = any(completed_mask & id_mask & time_mask)
        
        if viewed_and_completed:
            num_viewed_completed += 1
    
    # number of not viewed but completed offers:
    num_not_viewed_completed = num_completed - num_viewed_completed
    
    ##################################################################################
    #            STATS ONLY ABOUT THIS KIND OF OFFER
    ##################################################################################
    """This part is almost the same as the previous part, only at first the actual 
    transcript is reduced to only contain info which is related to an offer of this kind.
    """
    
    ############
    offer_id_mask = (actual_transcript.offer_id == offer.offer_id)
    actual_transcript = actual_transcript[offer_id_mask]
    ############
    
    if not actual_transcript.empty:
    
        # the number of received offers:
        received_mask = (actual_transcript.event == 'offer received')
        received_offers = actual_transcript[received_mask]
        num_received_this = received_mask.sum()

        # number of viewed offers:
        viewed_mask = (actual_transcript.event == 'offer viewed')
        num_viewed_this = viewed_mask.sum()

        # number of completed offers:
        completed_mask = (actual_transcript.event == 'offer completed')
        completed_offers = actual_transcript[completed_mask]
        num_completed_this = completed_mask.sum()

        # informational type offers can not be completed, so they are not included further in the viewed offers:
        info_ids = ['3f207df678b143eea3cee63160fa8bed', '5a8bc65990b245e5a138643cd4eb9837']
        not_info_mask = (actual_transcript.offer_id != info_ids[0]) & (actual_transcript.offer_id != info_ids[1])
        num_viewed_not_info_this = (viewed_mask & not_info_mask).sum()
        viewed_not_info_offers = actual_transcript[viewed_mask & not_info_mask]

        # number of viewed and completed offers:
        num_viewed_completed_this = 0
        for i, viewed_offer in viewed_not_info_offers.iterrows():
            offer_id = viewed_offer.offer_id
            time = viewed_offer.time
            duration = portfolio.duration[portfolio.id == offer_id].values[0]

            # beginning time of the viewed offer:
            # getting the last received offer of this type before it was viewed:
            mask_before = (received_offers.time <= time)
            mask_id = (received_offers.offer_id == offer_id)
            begin_time = received_offers.time[mask_before & mask_id].max()

            # expiration time of the viewed offer:
            expiration_time = begin_time + duration

            # checking if there is an offer after viewing and before expiration which was completed
            # and has the same offer_id
            id_mask = (completed_offers.offer_id == offer_id)
            time_mask = (completed_offers.time > time) & (completed_offers.time <= expiration_time)

            # viewed_and_completed is True if the offer was completed after viewed and False otherwise:
            viewed_and_completed = any(completed_mask & id_mask & time_mask)

            if viewed_and_completed:
                num_viewed_completed_this += 1

        # number of not viewed but completed offers:
        num_not_viewed_completed_this = num_completed_this - num_viewed_completed_this
    
    else:
        num_received_this = 0
        num_viewed_this = 0
        num_viewed_completed_this = 0
        num_viewed_not_info_this = 0
        num_not_viewed_completed_this = 0
    
    
    ###################################################################################
    # Filling up the DataFrame with the calculated stats:
    ###################################################################################
    
    # number of received offers until receiving the offer
    offer_stats.iloc[0, 0] = num_received
    
    # ratio of viewed / received offers until receiving the offer
    if num_received != 0:
        offer_stats.iloc[0, 1] = num_viewed / num_received
    else:
        offer_stats.iloc[0, 1] = 0
    
    # ratio of completed / viewed offers until receiving the offer
    if num_viewed_not_info != 0:
        offer_stats.iloc[0, 2] = num_viewed_completed / num_viewed_not_info
    else:
        offer_stats.iloc[0, 2] = 0
        
    # ratio of completed not viewed / received offers until receiving the offer
    if num_received != 0:
        offer_stats.iloc[0, 3] = num_not_viewed_completed / num_received
    else:
        offer_stats.iloc[0, 3] = 0
        
    # number of received offers of this type until receiving the offer
    offer_stats.iloc[0, 4] = num_received_this
        
    # ratio of viewed / received offers of this type until receiving the offer
    if num_received_this != 0:
        offer_stats.iloc[0, 5] = num_viewed_this / num_received_this
    else:
        offer_stats.iloc[0, 5] = 0
        
    # ratio of completed / viewed offers of this type until receiving the offer
    if num_viewed_not_info_this != 0:
        offer_stats.iloc[0, 6] = num_viewed_completed_this / num_viewed_not_info_this
    else:
        offer_stats.iloc[0, 6] = 0
        
    # ratio of completed / not viewed offers of this type until receiving the offer
    if num_received_this != 0:
        offer_stats.iloc[0, 7] = num_not_viewed_completed_this / num_received_this
    else:
        offer_stats.iloc[0, 7] = 0
    
    
    return offer_stats


def cathegorize_offer(viewed_offers_this, completed_offers_this):
    """Cathegorizes the offer into one of the following cathegories:
    0: not viewed, not completed
    1: not viewed but completed
    2: viewed but not completed
    3: viewed and completed
    """ 
    
    # checking, into which cathegory the offer falls:
    if not viewed_offers_this.empty:
        # the time the offer was viewed:
        viewed_time = viewed_offers_this.time.min()

        if not completed_offers_this.empty:
            # the time the offer was completed:
            completed_time = completed_offers_this.time.min()

            if viewed_time <= completed_time:
                # viewed and completed
                label = 3
            else:
                # completed before viewed
                label = 1
        else:
            # viewed but not completed:
            label= 2
    else:
        if not completed_offers_this.empty:
            # completed before viewed
            label = 1
        else:
            # not viewed nor completed:
            label = 0
    
    return label