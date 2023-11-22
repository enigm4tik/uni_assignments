/**
Description: Conway's Game of Life
Part 1: Calculate Game of Life in sequential mode on CPU
* Consider wrap-around
* Read input board from file
* Write output board to file 
* Measure setup, computation and finalization time 

Implement 4 command line parameters
 --load <filename>
 --save <filename>
 --generations <n>
 --measure

Rules 
 1) Dead cells with exactly 3 neighbors become alive
 2) Live cells with exactly 2 or 3 neighbors stay alive
 3) Live cells with less than 2 neighbors die
 4) Live cells with more than 3 neighbors die
 
 Ideas for improvement:
 Replace a number of for loops 
**/

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


int main(int argc, char ** argv)
{
  Timing* timing = Timing::getInstance();
  string filename_in, filename_out = "";
  
  int generations = 250;
  int measure = 0;
  int cols, rows = 0;
  
  static const struct option long_options[] = 
  {
    {"help", no_argument, 0, 'h'},
    {"load", required_argument, 0, 'l'},
    {"save", required_argument, 0, 's'},
    {"generations", required_argument, 0, 'g'},
    {"measure", no_argument, 0, 'm'},
    0
  };

  while (true) 
  {
    int index = -1;
    struct option * opt = 0;
    int result = getopt_long(argc, argv, "hml:s:g:", long_options, &index);
    
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
    for (int r=0; r < rows; r++) 
    {
      for (int c=0; c < cols; c++) 
      {
        int sum = 0;
        for (int row =-1; row < 2; row++)
        {
          for (int col =-1; col <2; col++) 
          {
            if (row == 0 && col == 0) continue; // I'm not my own neighbor
            sum += board[(row+r+rows)%rows][(col+cols+c)%cols];
            if (sum > 3) {
              board2[r][c] = 0;
              break;
            }
          }
        }
        
        if (sum < 2) 
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
