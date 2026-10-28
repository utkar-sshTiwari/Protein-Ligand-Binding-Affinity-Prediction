{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    torch
    rdkit
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    tqdm
    jupyter
    notebook
    ipykernel
    tensorboard
    graphviz
    pip
  ]);

  cppRuntime = pkgs.stdenv.cc.cc.lib;
in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
    pkgs.graphviz
    pkgs.git
    cppRuntime
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ cppRuntime ];
}
