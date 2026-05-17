<?php

class GeographClipModel extends GeographModelBase {
    public function predict_api($inputs, $threshold = 0.2) {
        $img_embeds = $inputs['clip-image'];
        $txt_embeds = $inputs['clip-title'];
        $image_id = $inputs['image_id'] ?? null;

        $x = $this->concat([$img_embeds, $txt_embeds]);

        // head.0, head.3
        $x = $this->linear($x, "head.0.weight", "head.0.bias");
        $x = $this->relu($x);
        $x = $this->linear($x, "head.3.weight", "head.3.bias");

        $probs = $this->sigmoid($x);

        $classes = $this->metadata['classes'];
        $flat_results = [];
        $found = false;

        foreach ($probs as $j => $prob) {
            if ($prob > $threshold) {
                $found = true;
                $flat_results[] = [
                    "image_id" => $image_id,
                    "model" => "clip",
                    "label" => $classes[$j] ?? "tag_$j",
                    "score" => (float)$prob
                ];
            }
        }

        if (!$found) {
            $flat_results[] = [
                "image_id" => $image_id,
                "model" => "clip",
                "label" => "None",
                "score" => 0.0
            ];
        }

        return $flat_results;
    }
}
