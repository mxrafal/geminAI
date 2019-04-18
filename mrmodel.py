from textgenrnn import textgenrnn  


def get_model_api():
    textgen = textgenrnn(weights_path='zodiac_weights.hdf5', vocab_path='zodiac_vocab.json', config_path='zodiac_config.json')
    
    textgen.generate()
    # textgen.generate(1, return_as_list=True)[0]

    def get_result():
        res = textgen.generate(1, return_as_list=True)[0]
        signs = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces']
        for s in signs:
            if s in res:
                res.replace(s, '')
        return res
        
    return get_result