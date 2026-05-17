<?php

abstract class GeographModelBase {
    protected $weights;
    protected $metadata;

    public function __construct($weightsPath) {
        if (!file_exists($weightsPath)) {
            throw new Exception("Weights file not found: $weightsPath");
        }
        $data = json_decode(file_get_contents($weightsPath), true);
        $this->weights = $data['state_dict'];
        $this->metadata = $data['metadata'];
    }

    abstract public function predict_api($inputs);

    protected function linear($input, $weightName, $biasName) {
        if (!isset($this->weights[$weightName]) || !isset($this->weights[$biasName])) {
             // Fallback for missing weights (could happen if model architecture doesn't match dummy weights)
             // But in our case we want to know if it's missing.
             throw new Exception("Weights or bias not found: $weightName, $biasName");
        }
        $weights = $this->weights[$weightName];
        $bias = $this->weights[$biasName];
        $output = [];

        $numRows = count($weights);
        $numCols = count($weights[0]);

        for ($i = 0; $i < $numRows; $i++) {
            $sum = $bias[$i];
            for ($j = 0; $j < $numCols; $j++) {
                if (isset($input[$j])) {
                    $sum += $input[$j] * $weights[$i][$j];
                }
            }
            $output[] = $sum;
        }
        return $output;
    }

    protected function relu($input) {
        return array_map(function($v) { return max(0, $v); }, $input);
    }

    protected function sigmoid($input) {
        return array_map(function($v) { return 1 / (1 + exp(-$v)); }, $input);
    }

    protected function softmax($input) {
        $max = max($input);
        $exps = array_map(function($v) use ($max) { return exp($v - $max); }, $input);
        $sum = array_sum($exps);
        return array_map(function($v) use ($sum) { return $v / $sum; }, $exps);
    }

    protected function embedding($idx, $weightName) {
        return $this->weights[$weightName][$idx];
    }

    protected function concat($vecs) {
        return array_merge(...$vecs);
    }
}
