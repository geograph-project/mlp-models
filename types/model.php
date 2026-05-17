<?php

class GeographTypesModel extends GeographModelBase {
    public function predict_api($inputs, $threshold = 0.5) {
        $clip_vec = $this->decode_embedding($inputs['clip-image']);
        $dist_idx = $this->get_dist_idx($inputs['distance']);
        $image_id = $inputs['image_id'] ?? null;

        $d_feat = $this->embedding($dist_idx, "dist_emb.weight");
        $x = $this->concat([$clip_vec, $d_feat]);

        // fc.0, fc.3
        $x = $this->linear($x, "fc.0.weight", "fc.0.bias");
        $x = $this->relu($x);
        $x = $this->linear($x, "fc.3.weight", "fc.3.bias");

        $probs = $this->sigmoid($x);
        $classes = $this->metadata['classes'];

        $flat_results = [];
        $matches = [];

        foreach ($probs as $j => $prob) {
            if ($prob > $threshold) {
                $matches[] = [$classes[$j], $prob];
            }
        }

        if (empty($matches)) {
            $max_val = -1.0;
            $top_idx = 0;
            foreach ($probs as $j => $prob) {
                if ($prob > $max_val) {
                    $max_val = $prob;
                    $top_idx = $j;
                }
            }
            $matches[] = [$classes[$top_idx] . " (Low)", $max_val];
        }

        foreach ($matches as $match) {
            $flat_results[] = [
                "image_id" => $image_id,
                "model" => "types",
                "label" => $match[0],
                "score" => (float)$match[1]
            ];
        }

        return $flat_results;
    }
}
