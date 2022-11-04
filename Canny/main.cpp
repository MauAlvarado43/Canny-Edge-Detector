// Alvarado L�pez Mauricio 5BM1
// Detector de bordes Canny

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#define PI 3.141596
#define vvd vector<vector<float>>

using namespace cv;
using namespace std;

// Funci�n para agregar bordes a la im�gen para que un kernel de nxn sea capaz de convolucionar
// con la im�gen
Mat padding(Mat& img, int n) {

	// Como el centro de la im�gen lo obtenemos con (n - 1) / 2, entonces 
	// habr�n (n - 1) / 2 pixeles hacia ambos lados, es decir, con este n�mero
	// conocemos cuantos pixeles extras habr�n hacia cada borde de la im�gen
	int mid = (n - 1) / 2;
	Mat p�dding = Mat::zeros(img.rows + (2 * mid), img.cols + (2 * mid), CV_8UC1);

	// Recorremos la im�gen y si nos encontramos fuera de los l�mites, dejamos 0 como valor
	for (int i = 0; i < p�dding.rows; i++) {
		for (int j = 0; j < p�dding.cols; j++) {

			if (i < mid || i > img.rows) continue;
			if (j < mid || j > img.cols) continue;

			// En otro caso, mapeamos con "mid" los pixeles de la im�gen original a 
			// la que tiene padding
			p�dding.at<uchar>(i, j) = img.at<uchar>(i - mid, j - mid);

		}
	}

	return p�dding;

}

// Funci�n que calcula el valor del elemento (x, y) del kernel para el filtro gaussiano
float gaussianFunction(float sigma, int x, int y) {
	return (1 / (2 * PI * pow(sigma, 2))) * exp(-((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))));
}

// Dado un tama�o n y un sigma, calculamos el kernel del filtro gaussiano
vvd gaussianKernel(int n, float sigma) {
	
	// Al ser un kernel cuadrado impar, el centro est� definido por (n - 1) / 2, as� que 
	// vamos a tomar los coordenadas partiendo desde ((n - 1) / 2, (n - 1) / 2)) para mantener 
	// el orden en el filtro
	vvd filter(n, vector<float>(n));
	int mid = (n - 1) / 2;
	float sum = 0;

	// Metemos los valores en la matriz y calculamos el m�ximo registrado,
	// esto para normalizar los niveles de brillo
	for(int gx = -mid; gx <= mid; gx++) 
		for (int gy = -mid; gy <= mid; gy++) {
			filter[gx + mid][gy + mid] = gaussianFunction(sigma, gx, gy);
			sum += filter[gx + mid][gy + mid];
		}

	// Imprimimos el kernel del filtro sin normalizado
	cout << "Kernel sin normalizar: \n";
	for (int i = 0; i < filter.size(); i++) {
		for (int j = 0; j < filter[i].size(); j++) {
			cout << filter[i][j] << "\t";
		}
		cout << "\n";
	}

	// Normalizamos
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			filter[i][j] /= sum;

	// Imprimimos el kernel del filtro ya normalizado
	cout << "Kernel normalizado: \n";
	for (int i = 0; i < filter.size(); i++) {
		for (int j = 0; j < filter[i].size(); j++) {
			cout << filter[i][j] << "\t";
		}
		cout << "\n";
	}

	return filter;

}

// Funci�n para realizar un filtro gaussiano
Mat gaussianFilter(Mat padding, int n, float sigma) {

	// Inicializamos una nueva im�gen, obtenemos el kernel gaussiano para n y calculamos mid
	Mat gauss = Mat::zeros(padding.rows, padding.cols, CV_8UC1);
	auto filter = gaussianKernel(n, sigma);
	int mid = (n - 1) / 2;

	// Generamos una nueva im�gen donde guardaremos la filtrada y le asignamos el tama�o de la original
	Mat filtered = Mat::zeros(padding.rows - (2 * mid), padding.cols - (2 * mid), CV_8UC1);

	// Realizamos la convoluci�n empezando en la im�gen desde (mid, mid) y asegurarnos
	// que no nos salimos de la im�gen. Adem�s, mapeamos el kernel con las coordenadas como deber�an ser 
	// para tomar los pixeles de la im�gen con relleno seg�n corresponda
	for (int i = mid; i < padding.rows - mid; i++)
		for (int j = mid; j < padding.cols - mid; j++)
			for (int k = -mid; k <= mid; k++)
				for (int l = -mid; l <= mid; l++)
					filtered.at<uchar>(i - mid, j - mid) += filter[k + mid][l + mid] * padding.at<uchar>(i + k, j + l);

	return filtered;

}

// Kernels Sobel para obtener los gradientes de la im�gen
vvd sobel_gx = {
	{-1, 0, 1} ,
	{-2, 0, 2},
	{-1, 0, 1}
};
vvd sobel_gy = {
	{1, 2, 1} ,
	{0, 0, 0},
	{-1, -2, -1}
};

// Funci�n para obtener el gradiente de la im�gen
tuple<Mat, Mat, Mat, Mat> gradient(Mat& img) {

	// Inicializamos las matrices, para el �ngulo la tomamos como float para evitar errores
	// en los c�lculos
	Mat gx = Mat::zeros(img.rows, img.cols, CV_8UC1);
	Mat gy = Mat::zeros(img.rows, img.cols, CV_8UC1);
	Mat gm = Mat::zeros(img.rows, img.cols, CV_8UC1);
	Mat ga = Mat::zeros(img.rows, img.cols, CV_32FC1);
	
	// Recorremos la im�gen evitando salirnos de ella
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {

			float x = 0;
			float y = 0;

			// Hacemos la convoluci�n de los kernels y la im�gen
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					x += sobel_gx[k + 1][l + 1] * img.at<uchar>(i + k, j + l);
					y += sobel_gy[k + 1][l + 1] * img.at<uchar>(i + k, j + l);
				}
			}

			// Asignamos los valores a su matriz seg�n corresponda
			gx.at<uchar>(i, j) = x;
			gy.at<uchar>(i, j) = y;
			gm.at<uchar>(i, j) = sqrt(pow(x, 2) + pow(y, 2));
			ga.at<float>(i, j) = atan2(y, x);

		}
	}
			
	return { gx, gy, gm, ga };

}

// Funci�n que realiza la supresi�n non-maximum
Mat nms(Mat gm, Mat ga) {

	// Inicializamos una matriz para guardar los valores del m�todo
	Mat nms_applied = Mat::zeros(gm.rows, gm.cols, CV_8UC1);

	// Recorremos la matriz sin salirnos de los bordes
	for (int i = 1; i < gm.rows - 1; i++) {
		for (int j = 1; j < gm.rows - 1; j++) {

			// Obtenemos el �ngulo del gradiente con valor absoluto para ahorrarnos 
			// comparaciones debajo
			float angle = abs(ga.at<float>(i, j));
			uchar p, q;

			// Si el �ngulo est� entre 0 y 22.5, o 157.5 y 180, entonces trabajaremos
			// con los pixeles a la izquierda y derecha
			if (0 <= angle < 22.5 || 157.5 <= angle <= 180) {
				p = gm.at<uchar>(i, j + 1);
				q = gm.at<uchar>(i, j - 1);
			}
			// Si el �ngulo est� entre 22.5 y 67.5, entonces trabajamos con los pixeles
			// arriba a la derecha y abajo a la izquierda
			else if (22.5 <= angle < 67.5) {
				p = gm.at<uchar>(i + 1, j - 1);
				q = gm.at<uchar>(i - 1, j + 1);
			}
			// Si el �ngulo est� entre 67.5 y 112.5, entonces tomamos los pixeles
			// de arriba y abajo
			else if (67.5 <= angle < 112.5) {
				p = gm.at<uchar>(i + 1, j);
				q = gm.at<uchar>(i - 1, j);
			}
			// Si el �ngulo est� entre 112.5 y 157.5, entonces tomamos los pixeles
			// de arriba a la izquierda y abajo a la derecha
			else if(112.5 <= angle < 157.5) {
				p = gm.at<uchar>(i - 1, j - 1);
				q = gm.at<uchar>(i + 1, j + 1);
			}

			// Si nuestro pixel es mayor que cualquiera de los otros 2, entonces mantenemos
			// el nivel de brillo, si no lo dejamos en 0
			if (gm.at<uchar>(i, j) >= q && gm.at<uchar>(i, j) >= p)
				nms_applied.at<uchar>(i, j) = gm.at<uchar>(i, j);

		}
	}

	return nms_applied;

}

// Funci�n que realiza hist�resis de la im�gen
Mat hysteresis(Mat nms_applied, int low_threshold, int high_threshold) {

	// Inicializamos una matriz
	Mat histeresys_applied = Mat::zeros(nms_applied.rows, nms_applied.cols, CV_8UC1);

	// Recorremos la im�gen
	for (int i = 0; i < nms_applied.rows; i++) {
		for (int j = 0; j < nms_applied.rows ; j++) {
			// Si el pixel est� por debajo del umbral bajo, entonces lo hacemos 0
			if (nms_applied.at<uchar>(i, j) <= low_threshold) 
				histeresys_applied.at<uchar>(i, j) = 0;
			// Si est� entre los 2 umbrales, le asignamos el m�nimo
			else if (nms_applied.at<uchar>(i, j) > low_threshold && nms_applied.at<uchar>(i, j) < high_threshold) 
				histeresys_applied.at<uchar>(i, j) = low_threshold;
			// Si es mayor que el umbra alto, entonces asignamos 255
			else if (nms_applied.at<uchar>(i, j) >= high_threshold) 
				histeresys_applied.at<uchar>(i, j) = 255;
		}
	}

	return histeresys_applied;

}

int main() {

	// Declaramos las variables para solicitarlas
	int n = 0;
	float sigma = 0;
	int low_threshold = 0;
	int high_threshold = 0;

	// Cargamos la im�gen
	char imageName[] = "lena.png";
	Mat image = imread(imageName);

	// Solicitamos y validamos las entradas
	while (n % 2 != 1 || n == 0) { cout << "Ingrese el valor de n: "; cin >> n; }
	while (sigma == 0) { cout << "Ingrese el valor de sigma: "; cin >> sigma; }
	cout << "Ingrese el umbral bajo: "; cin >> low_threshold;
	cout << "Ingrese el umbral alto: "; cin >> high_threshold;

	// Validamos que cargamos la im�gen
	if (!image.data) {
		cout << "Error al cargar la imagen: " << imageName << endl;
		exit(1);
	}

	// Convertimos la im�gen a escala de grises

	// Calculamos cada paso:
	// 1.- Escala de grises
	// 2.- Relleno (agregamos pixeles negros en bordes)
	// 3.- Suaviazado (aplicamos el filtro gaussiano)
	// 4.- Gradiente (aplicamos los operadores sobel)
	// 5.- NMS (adelgazar bordes)
	// 6.- Hysteresis (umbralizamos los bordes)

	// Convertimos a escala de grises
	Mat gray = Mat::zeros(image.rows, image.cols, CV_8UC1);
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Agregamos relleno y suavizamos
	Mat padd = padding(gray, n);
	Mat gauss = gaussianFilter(padd, n, sigma);

	// Aplicamos sobel para el gradiente
	auto sobel = gradient(gauss);
	Mat sobel_gx = get<0>(sobel);
	Mat sobel_gy = get<1>(sobel);
	Mat sobel_magnitude = get<2>(sobel);
	Mat sobel_angle = get<3>(sobel);

	// Aplicamos NMS e hist�resis
	Mat nms_applied = nms(sobel_magnitude, sobel_angle);
	Mat hysteresis_applied = hysteresis(nms_applied, low_threshold, high_threshold);

	// Imprimimos tama�os
	cout << "Size original: " << image.rows << "x" << image.cols << "\n";
	cout << "Size grises: " << gray.rows << "x" << gray.cols << "\n";
	cout << "Size relleno: " << padd.rows << "x" << padd.cols << "\n";
	cout << "Size suavizada: " << gauss.rows << "x" << gauss.cols << "\n";
	cout << "Sobel Gx: " << sobel_gx.rows << "x" << sobel_gx.cols << "\n";
	cout << "Sobel Gy: " << sobel_gy.rows << "x" << sobel_gy.cols << "\n";
	cout << "Sobel |G|: " << sobel_magnitude.rows << "x" << sobel_magnitude.cols << "\n";
	cout << "Sobel angulo: " << sobel_angle.rows << "x" << sobel_angle.cols << "\n";
	cout << "NMS: " << nms_applied.rows << "x" << nms_applied.cols << "\n";
	cout << "Histeresis: " << hysteresis_applied.rows << "x" << hysteresis_applied.cols << "\n";

	// Mostramos im�genes
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", image);

	namedWindow("Grises", WINDOW_AUTOSIZE);
	imshow("Grises", gray);

	namedWindow("Relleno", WINDOW_AUTOSIZE);
	imshow("Relleno", padd);

	namedWindow("Suavizada", WINDOW_AUTOSIZE);
	imshow("Suavizada", gauss);

	namedWindow("Sobel Gx", WINDOW_AUTOSIZE);
	imshow("Sobel Gx", sobel_gx);

	namedWindow("Sobel Gy", WINDOW_AUTOSIZE);
	imshow("Sobel Gy", sobel_gy);

	namedWindow("Sobel |G|", WINDOW_AUTOSIZE);
	imshow("Sobel |G|", sobel_magnitude);

	namedWindow("Sobel angulo", WINDOW_AUTOSIZE);
	imshow("Sobel angulo", sobel_angle);

	namedWindow("NMS", WINDOW_AUTOSIZE);
	imshow("NMS", nms_applied);

	namedWindow("Histeresis (final)", WINDOW_AUTOSIZE);
	imshow("Histeresis (final)", hysteresis_applied);

	waitKey(0);

	return 0;

}