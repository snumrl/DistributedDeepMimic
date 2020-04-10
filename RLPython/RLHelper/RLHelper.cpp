#include <boost/python/numpy.hpp>
#include <Eigen/Dense>

namespace p = boost::python;
namespace np = boost::python::numpy;

/* python helper */
template <typename T, typename U>
static Eigen::Matrix<T, -1, -1> npToEigen2d(np::ndarray np){
	int r = np.shape(0), c = np.shape(1);
	Eigen::Matrix<T, -1, -1> mat(r, c);
	U *data = reinterpret_cast<U *>(np.get_data());
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			mat(i, j) = data[i * c + j];
	return mat;
}

template <typename T, typename U>
static np::ndarray eigenTonp2d(const Eigen::Matrix<U, -1, -1> &mat){
	int r = mat.rows(), c = mat.cols();
	p::tuple shape = p::make_tuple(r, c);
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray np = np::empty(shape, dtype);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			np[i][j] = mat(i, j);
	return np;
}

template <typename T, typename U>
static Eigen::Matrix<T, -1, 1> npToEigen(np::ndarray np){
	int sz = np.shape(0);
	Eigen::Matrix<T, -1, 1> vec(sz);
	U *data = reinterpret_cast<U *>(np.get_data());
	for (int i = 0; i < sz; i++)
		vec[i] = data[i];
	return vec;
}

template <typename T, typename U>
static np::ndarray eigenTonp(const Eigen::Matrix<U, -1, 1> &vec){
	p::tuple shape = p::make_tuple(vec.size());
	np::dtype dtype = np::dtype::get_builtin<T>();
	np::ndarray np = np::empty(shape, dtype);
	for (int i = 0; i < vec.size(); i++)
		np[i] = vec[i];
	return np;
}

boost::python::tuple getGAE(np::ndarray values_np, np::ndarray reward_np, np::ndarray done_np, double gamma, double lamda){
	Eigen::VectorXd values = npToEigen<double, float>(values_np);
	Eigen::VectorXd reward = npToEigen<double, float>(reward_np);
	Eigen::VectorXi done = npToEigen<int, int>(done_np);
	int n = values.size();
	Eigen::VectorXd returns = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd advants = Eigen::VectorXd::Zero(n);

	double prev_return = 0., current_return = 0.;
	double prev_value = 0., current_value = 0.;
	double prev_advant = 0., current_advant = 0.;
	for (int i = n - 1; i >= 0; i--){
		prev_return = done[i] ? 0. : current_return;
		prev_advant = done[i] ? 0. : current_advant;
		prev_value = done[i] ? 0. : current_value;

		current_return = reward[i] + gamma * prev_return;
		current_advant = prev_advant * gamma * lamda + (reward[i] - values[i] + gamma * prev_value);
		current_value = values[i];

		returns[i] = current_return;
		advants[i] = current_advant;
	}
	advants = advants / sqrt(advants.squaredNorm() / n);
	return p::make_tuple(eigenTonp<float>(returns), eigenTonp<float>(advants));
}

BOOST_PYTHON_MODULE(libRLHelper){
	np::initialize();

	p::def("getGAE", getGAE);
}

int main(){}