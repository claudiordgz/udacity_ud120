#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error = predictions - net_worths
    n = int(len(error)*0.9)
    # Let's sort the errors and add the original index to know which one they are
    errors = sorted([(k, v[0]) for k, v in enumerate(error)], key=lambda x: x[1])[:n]
    for i in range(n):
        # use the original index to retrieve data from original arrays
        j = errors[i][0]
        bundle = ages[j], net_worths[j], predictions[j]
        cleaned_data.append(bundle)    

    return cleaned_data

