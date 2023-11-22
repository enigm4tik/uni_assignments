/**
Description: Conway's Game of Life
Part 2: Calculate Game of Life Multi Threaded mode on GPU using OpenMP

Implement two new command line parameters
  -- mode seq (sequential version from Exercise 1)
  -- mode omp --threads <NUM> (parallel OpenMP version with NUM threads)
**/

#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <getopt.h>
#include <set>
#include "Timing.h"
#include <algorithm>

using namespace std;

void sequential_calculation(int rows, int cols, vector<vector<int>> &board, vector<vector<int>> &board2) {
  for (int r=0; r < rows; r++) 
  {
    for (int c=0; c < cols; c++) 
    {
      int sum = 0;
      for (int row =-1; row < 2; row++)
      {
        for (int col =-1; col <2; col++) 
        {
          if (row == 0 && col == 0) continue; 
          sum += board[(row+r+rows)%rows][(col+cols+c)%cols];
        }
      }
      
      if (sum < 2 || sum > 3) 
      {
        board2[r][c] = 0;
      } else if (sum == 3) 
      {
        board2[r][c] = 1;
      } else if (sum == 2 && board[r][c] == 1) 
      {
        board2[r][c] = 1;
      } else 
      {
        board2[r][c] = 0;
      }
    }
  }
}

void parallel_calculation(int rows, int cols, vector<vector<int>> &board, vector<vector<int>> &board2, int threads) {
  #pragma omp parallel for num_threads(threads) collapse(2)
  for (int r=0; r < rows; r++) 
  {
    for (int c=0; c < cols; c++) 
    {
      int sum = 0;
      for (int row =-1; row < 2; row++)
      {
        for (int col =-1; col <2; col++) 
        {
          if (row == 0 && col == 0) continue; 
          sum += board[(row+r+rows)%rows][(col+cols+c)%cols];
        }
      }
      
      if (sum < 2 || sum > 3) 
      {
        board2[r][c] = 0;
      } else if (sum == 3) 
      {
        board2[r][c] = 1;
      } else if (sum == 2 && board[r][c] == 1) 
      {
        board2[r][c] = 1;
      } else 
      {
        board2[r][c] = 0;
      }
    }
  }
}


int main(int argc, char ** argv)
{
  Timing* timing = Timing::getInstance();
  string filename_in, filename_out = "";
  
  int generations = 250;
  int measure = 0;
  int cols, rows = 0;
  int threads = 8;
  string mode = "seq";
  
  static const struct option long_options[] = 
  {
    {"help", no_argument, 0, 'h'},
    {"load", required_argument, 0, 'l'},
    {"save", required_argument, 0, 's'},
    {"generations", required_argument, 0, 'g'},
    {"measure", no_argument, 0, 'm'},
    {"threads", required_argument, 0, 't'},
    {"mode", required_argument, 0, 'o'},
    0
  };

  while (true) 
  {
    int index = -1;
    struct option *opt = 0;
    int result = getopt_long(argc, argv, "hml:s:g:o:t:", long_options, &index);
    
    if (result == -1) break;

    switch (result)
    {
      case 'h': 
        {
          cout << "Game of Life - Usage" << endl;
          cout << "[-l, --load <filename>]: loads seed from <filename>" << endl;
          cout << "[-s, --save <filename>]: saves result to <filename>" << endl;
          cout << "[-g, --generations <amount>]: determines calculation of <amount> generations" << endl;
          cout << "[-m, --measure]: enables time measurement, optional" << endl;
          cout << "[-t, --threads]: determines number of threads, default 2" << endl;
          cout << "[-h, --help]: shows this help message" << endl;
        }
        break;
      case 'm': 
        {
          measure = 1;
        }
        break;
      case 'l':
        {
          filename_in = optarg;
        }
        break;
      case 's': 
        {
          filename_out = optarg;
        }
        break;
      case 'g':
        {
          generations = stoi(optarg);
        }
        break;
      case 'o':
        {
          mode = optarg;
        }
        break;
      case 't':
        {
          threads = stoi(optarg);
        }
        break;
    }
  }


  /* Preparation */
  timing->startSetup();

  ifstream file(filename_in);
  string line;
  std::getline(file, line);
  stringstream ss(line);
  string token;
  std::getline(ss, token, ',');
  cols = std::stoi(token);
  std::getline(ss, token, ',');
  rows = std::stoi(token);

  vector<vector<int>> board(rows, std::vector<int>(cols, 0));
  
  char cell;
  for (int r = 0; r < rows; r++) 
  {
    for (int c = 0; c < cols; c++) 
    {
      file >> cell;
      if (cell == 'x') 
      {
        board[r][c] = 1;
      }
    }
  }
  
  vector<vector<int>> board2(rows, std::vector<int>(cols, 0));
  timing->stopSetup();
  
  /* Calculation */
  timing->startComputation();
  
  for (int g=0; g < generations; g++) 
  {
    if (mode == "omp") {
      parallel_calculation(rows, cols, board, board2, threads);
    } else {
      sequential_calculation(rows, cols, board, board2);
    }
    board.swap(board2);
  }

  timing->stopComputation();
  
  /* Finalization */
  timing->startFinalization();
  
  ofstream output_file;
  output_file.open(filename_out);
  output_file << cols << "," << rows << endl;
  for (int i = 0; i<rows; i++)
  {
    for (int j = 0; j < cols; j++) 
    {
      if (board[i][j] == 1)
      {
        output_file << "x";
      } else {
        output_file << ".";
      }   
    }
    output_file << "\n";
  }
  output_file.close();
  timing->stopFinalization();

  if (measure) {
    std::string output = timing->getResults();
    cout << output << endl;
  }
}
