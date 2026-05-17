<?php

class GeographSubjectModel extends GeographModelBase {
    public function predict_api($inputs, $k = 10) {
        $clip_vec = $inputs['clip-image'];
        $dist_idx = $inputs['distance'];
        $image_id = $inputs['image_id'] ?? null;

        $d_feat = $this->embedding($dist_idx, "dist_emb.weight");
        $x = $this->concat([$clip_vec, $d_feat]);

        $hidden_layers = $this->metadata['hidden_layers'];
        $layer_idx = 0;
        foreach ($hidden_layers as $h_dim) {
            $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");
            $x = $this->relu($x);
            $layer_idx += 3;
        }

        $x = $this->linear($x, "fc.$layer_idx.weight", "fc.$layer_idx.bias");

        $probs = $this->softmax($x);

        $names = $this->metadata['metadata']['names'];

        // Get top k
        arsort($probs);
        $flat_results = [];
        $count = 0;
        foreach ($probs as $idx => $prob) {
            if ($count >= $k) break;
            $flat_results[] = [
                "image_id" => $image_id,
                "model" => "subjects",
                "label" => $names[$idx],
                "score" => (float)$prob
            ];
            $count++;
        }

        return $flat_results;
    }
}
