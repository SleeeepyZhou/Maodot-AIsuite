#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

#include "ai_object.h"
#include "sd/stable-diffusion.h"

class StableDiffusion : public AIObject {
	GDCLASS(StableDiffusion, AIObject);

public:
	enum rng_type_t {
		STD_DEFAULT_RNG,
		CUDA_RNG
	};

	enum SamplerMethod {
		EULER_A,
		EULER,
		HEUN,
		DPM2,
		DPMPP2S_A,
		DPMPP2M,
		DPMPP2Mv2,
		IPNDM,
		IPNDM_V,
		LCM,
		N_SAMPLE_METHODS
	};

	enum Scheduler {
		DEFAULT,
		DISCRETE,
		KARRAS,
		EXPONENTIAL,
		AYS,
		GITS,
		N_SCHEDULES
	};

	// same as enum ggml_type
	enum sd_type_t {
		SD_TYPE_F32 = 0,
		SD_TYPE_F16 = 1,
		SD_TYPE_Q4_0 = 2,
		SD_TYPE_Q4_1 = 3,
		// SD_TYPE_Q4_2 = 4, support has been removed
		// SD_TYPE_Q4_3 = 5, support has been removed
		SD_TYPE_Q5_0 = 6,
		SD_TYPE_Q5_1 = 7,
		SD_TYPE_Q8_0 = 8,
		SD_TYPE_Q8_1 = 9,
		SD_TYPE_Q2_K = 10,
		SD_TYPE_Q3_K = 11,
		SD_TYPE_Q4_K = 12,
		SD_TYPE_Q5_K = 13,
		SD_TYPE_Q6_K = 14,
		SD_TYPE_Q8_K = 15,
		SD_TYPE_IQ2_XXS = 16,
		SD_TYPE_IQ2_XS = 17,
		SD_TYPE_IQ3_XXS = 18,
		SD_TYPE_IQ1_S = 19,
		SD_TYPE_IQ4_NL = 20,
		SD_TYPE_IQ3_S = 21,
		SD_TYPE_IQ2_S = 22,
		SD_TYPE_IQ4_XS = 23,
		SD_TYPE_I8 = 24,
		SD_TYPE_I16 = 25,
		SD_TYPE_I32 = 26,
		SD_TYPE_I64 = 27,
		SD_TYPE_F64 = 28,
		SD_TYPE_IQ1_M = 29,
		SD_TYPE_BF16 = 30,
		SD_TYPE_Q4_0_4_4 = 31,
		SD_TYPE_Q4_0_4_8 = 32,
		SD_TYPE_Q4_0_8_8 = 33,
		SD_TYPE_COUNT,
	};

private:
	int count;

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();
	void add(int p_value);
	void reset();
	int get_total() const;
};

#endif // STABLE_DIFFUSION_H



private:
	bool requesting = false;

	String request_string;
	String url;
	int port = 80;
	Vector<String> headers;
	bool use_tls = false;
	Ref<TLSOptions> tls_options;
	HTTPClient::Method method;
	Vector<uint8_t> request_data;

	bool request_sent = false;
	Ref<HTTPClient> client;
	PackedByteArray body;
	SafeFlag use_threads;
	bool accept_gzip = true;

	bool got_response = false;
	int response_code = 0;
	Vector<String> response_headers;

	String download_to_file;

	Ref<StreamPeerGZIP> decompressor;
	Ref<FileAccess> file;

	int body_len = -1;
	SafeNumeric<int> downloaded;
	SafeNumeric<int> final_body_size;
	int body_size_limit = -1;

	int redirections = 0;

	bool _update_connection();

	int max_redirects = 8;

	double timeout = 0;

	void _redirect_request(const String& p_new_url);

	bool _handle_response(bool* ret_value);

	Error _parse_url(const String& p_url);
	Error _request();

	bool has_header(const PackedStringArray& p_headers, const String& p_header_name);
	String get_header_value(const PackedStringArray& p_headers, const String& header_name);

	SafeFlag thread_done;
	SafeFlag thread_request_quit;

	Thread thread;

	void _defer_done(int p_status, int p_code, const PackedStringArray& p_headers, const PackedByteArray& p_data);
	void _request_done(int p_status, int p_code, const PackedStringArray& p_headers, const PackedByteArray& p_data);
	static void _thread_func(void* p_userdata);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Error request(const String& p_url, const Vector<String>& p_custom_headers = Vector<String>(), HTTPClient::Method p_method = HTTPClient::METHOD_GET, const String& p_request_data = ""); //connects to a full url and perform request
	Error request_raw(const String& p_url, const Vector<String>& p_custom_headers = Vector<String>(), HTTPClient::Method p_method = HTTPClient::METHOD_GET, const Vector<uint8_t>& p_request_data_raw = Vector<uint8_t>()); //connects to a full url and perform request
	void cancel_request();
	HTTPClient::Status get_http_client_status() const;

	void set_use_threads(bool p_use);
	bool is_using_threads() const;

	void set_accept_gzip(bool p_gzip);
	bool is_accepting_gzip() const;

	void set_download_file(const String& p_file);
	String get_download_file() const;

	void set_download_chunk_size(int p_chunk_size);
	int get_download_chunk_size() const;

	void set_body_size_limit(int p_bytes);
	int get_body_size_limit() const;

	void set_max_redirects(int p_max);
	int get_max_redirects() const;

	Timer* timer = nullptr;

	void set_timeout(double p_timeout);
	double get_timeout();

	void _timeout();

	int get_downloaded_bytes() const;
	int get_body_size() const;

	void set_http_proxy(const String& p_host, int p_port);
	void set_https_proxy(const String& p_host, int p_port);

	void set_tls_options(const Ref<TLSOptions>& p_options);

	HTTPRequest();
};

VARIANT_ENUM_CAST(HTTPRequest::Result);

#endif // HTTP_REQUEST_H
