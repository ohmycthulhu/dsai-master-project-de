#include <cstdlib>
#include <iostream>

using namespace std;

int main() {
    const size_t popSize = 192;

    float m_costs[popSize];
    size_t m_best_indices[popSize];
    size_t m_best_indices_tmp[popSize];

    srand(time(NULL));
    // Populate shared memory
    for (int idx = 0; idx < popSize; idx++) {
        m_costs[idx] = rand() % 1000;
        m_best_indices[idx] = idx;
    }


    for (size_t i = popSize; i > 1; i = i / 2 + i % 2) {
        cout << "Starting: " << i << endl;
        for (int idx = 0; idx < popSize; idx++) {
            if (idx >= (i + i % 2) / 2) {
                continue;
            }

            size_t t_1 = idx * 2, t_2 = t_1 + 1;
            if (t_2 >= i) {
                m_best_indices_tmp[idx] = t_1;
            } else {
                m_best_indices_tmp[idx] = m_costs[m_best_indices[t_1]] <= m_costs[m_best_indices[t_2]] ? t_1 : t_2;
            }
        }

        cout << "Finishing: " << i << endl;
        for (int idx = 0; idx < popSize; idx++) {
            if (idx >= (i + i % 2) / 2) {
                continue;
            }
            cout << "Element: " << idx << " - " << m_best_indices_tmp[idx] << endl;
            m_best_indices[idx] = m_best_indices[m_best_indices_tmp[idx]];
        }
        cout << "Finished: " << i << endl;
    }

    cout << "Best index is " << m_costs[m_best_indices[0]] << endl;
    cout << "Best index is " << m_best_indices[0] << endl;

    return 0;
}
