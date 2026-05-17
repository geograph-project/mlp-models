<?php

class GeographGalleryModel extends GeographModelBase {
    public function predict_api($inputs) {
        $clip_vec = $inputs['clip-image'];
        $pe_vec = $inputs['pe-image'];
        $image_id = $inputs['image_id'] ?? null;

        $x = $this->concat([$clip_vec, $pe_vec]);

        // Layers
        // PyTorch names: fc.0, fc.3, fc.6 ...
        // Depending on hidden layers.
        // For default [256]: 0 is Linear, 1 is ReLU, 2 is Dropout, 3 is Linear(output)

        $hidden_layers = $this->metadata['hidden_layers'];
        $layer_idx = 0;
        foreach ($hidden_layers as $h_dim) {
            $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");
            $x = $this->relu($x);
            $layer_idx += 3; // linear, relu, dropout
        }

        $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");

        $results = $x; // Multi-target regression [baysian, score]

        return [
            [
                "image_id" => $image_id,
                "model" => "baysian",
                "score" => $results[0] * 5.0,
            ],
            [
                "image_id" => $image_id,
                "model" => "score",
                "score" => $results[1] * 10.0,
            ]
        ];
    }
}
