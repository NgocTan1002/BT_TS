{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "290921d7-38f9-464d-b2ba-d1e6d61cb742",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Chuỗi quan sát: ['Umbrella', 'Umbrella', 'No Umbrella']\n",
      "Xác suất chuỗi quan sát (Forward): 0.1362\n",
      "Chuỗi trạng thái ẩn tốt nhất (Viterbi): ['Rainy', 'Rainy', 'Sunny']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# ===== 1. Khởi tạo mô hình =====\n",
    "\n",
    "states = ['Rainy', 'Sunny']\n",
    "observations = ['Umbrella', 'No Umbrella']\n",
    "state_dict = {state: i for i, state in enumerate(states)}\n",
    "obs_dict = {obs: i for i, obs in enumerate(observations)}\n",
    "\n",
    "# Xác suất khởi đầu\n",
    "start_prob = np.array([0.6, 0.4])\n",
    "\n",
    "# Ma trận chuyển trạng thái (from -> to)\n",
    "trans_prob = np.array([\n",
    "    [0.7, 0.3],  # Rainy -> [Rainy, Sunny]\n",
    "    [0.4, 0.6]   # Sunny -> [Rainy, Sunny]\n",
    "])\n",
    "\n",
    "# Ma trận xác suất phát xạ (state -> observation)\n",
    "emission_prob = np.array([\n",
    "    [0.9, 0.1],  # Rainy -> [Umbrella, No Umbrella]\n",
    "    [0.2, 0.8]   # Sunny -> [Umbrella, No Umbrella]\n",
    "])\n",
    "\n",
    "# Chuỗi quan sát (ví dụ: Umbrella, Umbrella, No Umbrella)\n",
    "obs_seq = ['Umbrella', 'Umbrella', 'No Umbrella']\n",
    "obs_idx = [obs_dict[o] for o in obs_seq]\n",
    "T = len(obs_idx)\n",
    "N = len(states)\n",
    "\n",
    "# ===== 2. Thuật toán Forward (đánh giá xác suất chuỗi quan sát) =====\n",
    "\n",
    "def forward_algo():\n",
    "    alpha = np.zeros((T, N))\n",
    "    # Khởi tạo\n",
    "    alpha[0] = start_prob * emission_prob[:, obs_idx[0]]\n",
    "    # Đệ quy\n",
    "    for t in range(1, T):\n",
    "        for j in range(N):\n",
    "            alpha[t, j] = np.sum(alpha[t-1] * trans_prob[:, j]) * emission_prob[j, obs_idx[t]]\n",
    "    return np.sum(alpha[-1]), alpha\n",
    "\n",
    "# ===== 3. Thuật toán Viterbi (giải mã chuỗi trạng thái ẩn tốt nhất) =====\n",
    "\n",
    "def viterbi_algo():\n",
    "    delta = np.zeros((T, N))\n",
    "    psi = np.zeros((T, N), dtype=int)\n",
    "    # Khởi tạo\n",
    "    delta[0] = start_prob * emission_prob[:, obs_idx[0]]\n",
    "    # Đệ quy\n",
    "    for t in range(1, T):\n",
    "        for j in range(N):\n",
    "            seq_probs = delta[t-1] * trans_prob[:, j]\n",
    "            psi[t, j] = np.argmax(seq_probs)\n",
    "            delta[t, j] = np.max(seq_probs) * emission_prob[j, obs_idx[t]]\n",
    "    # Truy vết lại đường đi\n",
    "    path = np.zeros(T, dtype=int)\n",
    "    path[-1] = np.argmax(delta[-1])\n",
    "    for t in reversed(range(1, T)):\n",
    "        path[t-1] = psi[t, path[t]]\n",
    "    state_seq = [states[i] for i in path]\n",
    "    return state_seq, delta\n",
    "\n",
    "# ===== 4. Chạy và in kết quả =====\n",
    "\n",
    "prob, alpha = forward_algo()\n",
    "state_seq, delta = viterbi_algo()\n",
    "\n",
    "print(\"Chuỗi quan sát:\", obs_seq)\n",
    "print(\"Xác suất chuỗi quan sát (Forward):\", round(prob, 4))\n",
    "print(\"Chuỗi trạng thái ẩn tốt nhất (Viterbi):\", state_seq)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98922fa6-2f2e-4504-9cbe-173767b9ecec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
