<?php

class GeographScenicModel extends GeographModelBase {
    public function predict_api($inputs) {
        $clip_vec = $inputs['clip-image'];
        $pe_vec = $inputs['pe-image'];
        $image_id = $inputs['image_id'] ?? null;

        $x = $this->concat([$clip_vec, $pe_vec]);

        $hidden_layers = $this->metadata['hidden_layers'];
        $layer_idx = 0;
        foreach ($hidden_layers as $h_dim) {
            $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");
            $x = $this->relu($x);
            $layer_idx += 3;
        }

        $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");

        $results = $x;

        return [
            [
                "image_id" => $image_id,
                "model" => "v_bayesian",
                "score" => $results[0] * 10.0,
            ]
        ];
    }
}
