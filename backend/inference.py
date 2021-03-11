import yaml
import torch
from preprocess import Preprocess
from bengali_model import BengaliLSTMAttentionClassifier
from hindi_model import SentimentNet
from bengali_model_config import config_dict as bengali_config_dict
from bengali_model_config import config_dict as hindi_config_dict
import copy
import pickle
from torch.autograd import Variable


class InferSentiment:
    """
    Class for loading model and making sentiment predictions
    """
    def __init__(self, lang, text):
        self.lang = lang
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.read_config()
        self.load_artefacts()
        self.text = Preprocess(self.inference_config['stopwords']).perform_preprocessing(text)        
    
    def load_artefacts(self):
        """
        Loads saved models and language vocabularies based on request language
        """
        try:
            with open(self.inference_config['vocab'], 'rb') as handle:
                vocab_items = pickle.load(handle)
                self.vocabulary = vocab_items['vocab']
                self.index2word = vocab_items['index2word']
                self.word2index = vocab_items['word2index']
                self.init_model()   
      
        except Exception as e:
            print(e)
    
    def predict(self):
        """
        Predicts the sentiment of the text and returns it 
        alongside attention weights
        """
        sequence_tensor, lengths_tensor = self.process_text()        
        pred, annotation_weight_matrix = None, None
        if self.lang == 'hindi':
            pred, annotation_weight_matrix = self.model(sequence_tensor, lengths_tensor, lang=self.lang)
        else:
            pred, annotation_weight_matrix = self.model(sequence_tensor, lengths_tensor)

        predictions = torch.max(pred, 1)[1].float() ## get the prediction values
        attention_weights = torch.mean(annotation_weight_matrix, dim=1) ## averaged attention weights

        ## create response dictionary
        response = {
            "prediction": int(predictions.cpu().numpy()[0]), 
            "attention_weights": attention_weights.detach().cpu().numpy()[0].tolist(),
            "clean_text": self.text
            }

        return response

    def process_text(self):
        """
        Vectorizes the text sequence, converts it to a tensor and also
        computes its length
        """
        vectorized_sequence = self.vectorize_sequence(self.text)
        sequence_length = len(vectorized_sequence) ## computing sequence lengths for padding
        vectorized_sequences = torch.LongTensor([vectorized_sequence]).permute(1, 0)
        length_tensor = torch.LongTensor([sequence_length]) ## casting to long 

        return torch.autograd.Variable(vectorized_sequences).to(self.device), length_tensor.cpu().numpy()

    def read_config(self, config_path='config.yaml'):
        """
        Reads configuration file and loads it to memory
        """
        with open(r'{}'.format(config_path)) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.inference_config = yaml.load(file, Loader=yaml.FullLoader)[self.lang]
    
    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        return [self.word2index[token] for token in sentence.split() if token in self.vocabulary] 
    
    def pad_sequences(self, vectorized_sequences, sequence_lengths):
        """ 
        Pads zeros at the end of each sequence in data tensor till max 
        length of sequence in that batch
        """
        padded_sequence_tensor = torch.zeros((len(vectorized_sequences), sequence_lengths.max())).long() ## init zeros tensor
        for idx, (seq, seqlen) in enumerate(zip(vectorized_sequences, sequence_lengths)): ## iterate over each sequence
           padded_sequence_tensor[idx, :seqlen] = torch.LongTensor(seq) ## each sequence get padded by zeros until max length in that batch
        return padded_sequence_tensor ## returns padded tensor

    def init_model(self):
        """
        Initializes models and loads them memory based on requested 
        language
        """
        if self.lang == 'bengali':
            config_dict = bengali_config_dict
            self.model = BengaliLSTMAttentionClassifier(batch_size=config_dict['batch_size'], output_size=config_dict['num_classes'], 
                            vocab_size=len(self.vocabulary), hidden_size=config_dict['hidden_size'], 
                            embedding_size=config_dict['embedding_size'], weights=torch.FloatTensor([]),
                            lstm_layers=config_dict['lstm_layers'], device=config_dict['device'],
                            bidirectional=config_dict['is_bi_lstm'], pretrained_path=self.inference_config['model'],
                            self_attention_config=config_dict['self_attention_config'], fc_hidden_size=config_dict['fc_hidden_size']).to(config_dict['device'])
            
            self.model.load_state_dict(copy.deepcopy(torch.load(self.inference_config['model'], self.device)))
            self.model.eval()
        else:
            config_dict = hindi_config_dict
            self.model = SentimentNet(batch_size=config_dict['batch_size'], output_size=config_dict['num_classes'], 
                            bengali_vocab_size=16622, hidden_size=config_dict['hidden_size'], 
                            embedding_size=config_dict['embedding_size'], hindi_weights=torch.FloatTensor([]), bengali_weights=torch.FloatTensor([]),
                            lstm_layers=config_dict['lstm_layers'], device=config_dict['device'], hindi_vocab_size=len(self.vocabulary),
                            bidirectional=config_dict['is_bi_lstm'], pretrained_path=None,
                            self_attention_config=config_dict['self_attention_config'], fc_hidden_size=config_dict['fc_hidden_size']).to(config_dict['device'])
            self.model.load_state_dict(copy.deepcopy(torch.load(self.inference_config['model'], self.device)))

    