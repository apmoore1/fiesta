{
  "dataset_reader": {
    "type": "target_dataset",
    "token_indexers": {
      "tokens": {
          "type": "single_id",
          "lowercase_tokens": true
      }
    }
  },
  "model": {
    "type": "interactive_attention_target_classifier",
    "text_field_embedder": {
      "token_embedders": {
          "tokens": {
              "type": "embedding",
              "embedding_dim": 300,
              "pretrained_file": "http://ucrel-web.lancaster.ac.uk/moorea/glove_vectors/semeval_restaurant_specific/restaurant_glove.840B.300d.txt",
              "trainable": true
          }
      }
    },
    "target_field_embedder": {
      "token_embedders": {
          "tokens": {
              "type": "embedding",
              "embedding_dim": 300,
              "pretrained_file": "http://ucrel-web.lancaster.ac.uk/moorea/glove_vectors/semeval_restaurant_specific/restaurant_glove.840B.300d.txt",
              "trainable": true
          }
      }
    },
    "text_encoder": {
      "type": "lstm",
      "bidirectional": false,
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "dropout": 0.0
    },
    "target_encoder": {
      "type": "lstm",
      "bidirectional": false,
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "dropout": 0.0
    },
    "dropout": 0.5,
    "word_dropout": 0.0
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "cuda_device": 0,
    "shuffle": true,
    "validation_metric": "+accuracy",
    "optimizer": {
        "type": "adam"
    }
  }
}