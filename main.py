from src import Parameters
from src.plot import Plot
from src.graph import  Graph
from src.gnn import run_gnn
from src.hpo import parse_hpos
from src.primekg import parse_primekg
from src.chapter_data import parse_icd
from src.phecodes import parse_phecodes


def main():
    # load model parameters
    params = Parameters()

    # parse icd codes
    icd = parse_icd(file_list=params.icd_files)
    # parse phecode to icd file
    phecodes2icd = parse_phecodes(file_name=params.phecode_file)
    # parse hpo to phecode file
    hpo2phecode = parse_hpos(file_name=params.hpo_file,
                             subset_phecodes=phecodes2icd.phecodes)
    primekg = parse_primekg(file_name=params.primekg_file,
                            edge_types=params.edge_type_list)

    # generate graphs
    graph = Graph(p=params,
                  icd_data=icd,
                  phecode_data=phecodes2icd,
                  hpo_data=hpo2phecode,
                  primekg_data=primekg)

    # save graph statistiscs
    graph.graph_statistics()

    # Graph Neural Network training
    results = run_gnn(
        graph=graph.torch_graph,
        edge_type=params.edge_type,
        epochs=params.epochs,
        itta=params.learning_rate
    )

    # plotting
    plot = Plot()
    plot.interactive_graph(g=graph,
                           outfile=params.interactive_plot_file)
    plot.embedding_space(emb=results.training_embedding_space,
                         parameters=params,
                         title='Training set embeddings',
                         outfile=params.training_embedding_plot_file)
    #plot.graph(g=graph,
    #           outfile=params.plot_file)


if __name__ == '__main__':
    main()
